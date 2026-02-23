use super::BufferUsages;
use crate::ShaderArgs;
use crate::backend::{
    Backend, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, ShaderBinding,
};
use crate::shader::ShaderArgsError;
use bytemuck::{AnyBitPattern, NoUninit};
use encase::{ShaderType, StorageBuffer};
use minislang::shader_slang;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};

// Slang's ComputeVaryingInput structure for host-callable kernels
// This matches the layout Slang expects for thread indexing
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ComputeVaryingInput {
    start_thread_id: [u32; 3],
    end_thread_id: [u32; 3],
}

// Thread-safe wrapper for raw pointers
struct SendPtr(*mut std::ffi::c_void);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

// Helper trait to extract raw pointers from any buffer type
trait BufferPointer: Send + Sync {
    fn as_ptr(&self) -> *mut std::ffi::c_void;
}

impl<T: DeviceValue> BufferPointer for Arc<Mutex<CpuBuffer<T>>> {
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        let buf = self.lock().unwrap();
        buf.data.as_ptr() as *mut std::ffi::c_void
    }
}

/// CPU backend for executing compute shaders on the CPU.
///
/// This backend loads shared libraries compiled from shader code and executes
/// them as native CPU functions.
pub struct Cpu {
    /// Loaded libraries (kept alive)
    libraries: Arc<Mutex<Vec<libloading::Library>>>,
}

impl Cpu {
    /// Creates a new CPU backend instance.
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            libraries: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

/// CPU buffer wrapper - just a Vec wrapper.
pub struct CpuBuffer<T: DeviceValue> {
    data: Vec<T>,
    usage: BufferUsages,
}

impl<T: DeviceValue> CpuBuffer<T> {
    fn new_with_data(data: Vec<T>, usage: BufferUsages) -> Self {
        Self { data, usage }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// CPU buffer slice - just references the underlying buffer.
pub struct CpuBufferSlice<T: DeviceValue> {
    buffer: Arc<Mutex<CpuBuffer<T>>>,
    start: usize,
    end: usize,
}

impl<T: DeviceValue> Clone for CpuBufferSlice<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            start: self.start,
            end: self.end,
        }
    }
}

/// CPU command encoder - batches operations.
pub struct CpuEncoder {
    operations: Vec<CpuOperation>,
}

enum CpuOperation {
    #[allow(dead_code)]
    CopyBuffer {
        src: Arc<dyn std::any::Any + Send + Sync>,
        dst: Arc<dyn std::any::Any + Send + Sync>,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
        element_size: usize,
    },
}

impl CpuEncoder {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    fn finish(self) -> Vec<CpuOperation> {
        self.operations
    }
}

/// CPU compute pass.
pub struct CpuPass {
    _operations: Arc<Mutex<Vec<CpuOperation>>>,
}

/// CPU module - a loaded shared library.
pub struct CpuModule {
    library: Arc<libloading::Library>,
}

/// CPU function - references to the entry point in the loaded library.
pub struct CpuFunction {
    library: Arc<libloading::Library>,
    entry_point: String,
}

/// CPU dispatch state - collects arguments before execution.
pub struct CpuDispatch<'a> {
    function: &'a CpuFunction,
    bindings: HashMap<u32, Arc<dyn BufferPointer>>,
}

#[derive(thiserror::Error, Debug)]
pub enum CpuBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error("CPU error: {0}")]
    Cpu(String),
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    #[error("Library loading error: {0}")]
    LibLoading(#[from] libloading::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[async_trait::async_trait]
impl Backend for Cpu {
    const NAME: &'static str = "cpu";
    const TARGET: super::CompileTarget = super::CompileTarget::HostHostCallable;

    type Error = CpuBackendError;
    type Buffer<T: DeviceValue> = Arc<Mutex<CpuBuffer<T>>>;
    type BufferSlice<'b, T: DeviceValue> = CpuBufferSlice<T>;
    type Encoder = CpuEncoder;
    type Pass = CpuPass;
    type Module = CpuModule;
    type Function = CpuFunction;
    type Dispatch<'a> = CpuDispatch<'a>;

    fn as_cpu(&self) -> Option<&Cpu> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        // Write bytes to a temporary file and load as shared library
        let temp_dir = std::env::temp_dir();

        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        let lib_path = temp_dir.join(format!("slang_cpu_{}.so", uuid::Uuid::new_v4()));

        #[cfg(target_os = "macos")]
        let lib_path = temp_dir.join(format!("slang_cpu_{}.dylib", uuid::Uuid::new_v4()));

        #[cfg(target_os = "windows")]
        let lib_path = temp_dir.join(format!("slang_cpu_{}.dll", uuid::Uuid::new_v4()));

        std::fs::write(&lib_path, bytes)?;

        let library = unsafe { libloading::Library::new(&lib_path)? };

        // Keep the library alive - store before wrapping in Arc
        let library_copy = unsafe { libloading::Library::new(&lib_path)? };
        self.libraries.lock().unwrap().push(library_copy);

        let library = Arc::new(library);

        Ok(CpuModule { library })
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error> {
        Ok(CpuFunction {
            library: module.library.clone(),
            entry_point: entry_point.to_string(),
        })
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        CpuEncoder::new()
    }

    fn begin_dispatch<'a>(
        &'a self,
        _pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        CpuDispatch {
            function,
            bindings: HashMap::new(),
        }
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        let operations = encoder.finish();

        // Execute all buffered operations
        for op in operations {
            match op {
                CpuOperation::CopyBuffer {
                    src: _,
                    dst: _,
                    src_offset: _,
                    dst_offset: _,
                    len: _,
                    element_size: _,
                } => {
                    // This is a simplified version - in practice we'd need type erasure
                    // that preserves the ability to access the underlying data
                    // For now, we'll skip the actual copy as it's complex with type erasure
                }
            }
        }

        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // CPU execution is synchronous, so nothing to do
        Ok(())
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(Arc::new(Mutex::new(CpuBuffer::new_with_data(
            data.to_vec(),
            usage,
        ))))
    }

    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        // For CPU backend, we'll store the encased data as raw bytes in a Vec<T>
        // This is a simplification - ideally we'd have a separate encased buffer type
        let mut bytes = vec![];
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        // Convert bytes to T vector
        let num_ts = bytes.len().div_ceil(std::mem::size_of::<T>());
        let mut result = Vec::with_capacity(num_ts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
            result.set_len(num_ts);
        }

        Ok(Arc::new(Mutex::new(CpuBuffer::new_with_data(
            result, usage,
        ))))
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        // Create uninitialized buffer - use zeroed memory
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
            std::ptr::write_bytes(data.as_mut_ptr(), 0, len);
        }
        Ok(Arc::new(Mutex::new(CpuBuffer::new_with_data(data, usage))))
    }

    fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let size = T::min_size().get() as usize * len;
        let size_in_t = size.div_ceil(std::mem::size_of::<T>());
        let mut data = Vec::with_capacity(size_in_t);
        unsafe {
            data.set_len(size_in_t);
            std::ptr::write_bytes(data.as_mut_ptr(), 0, size_in_t);
        }
        Ok(Arc::new(Mutex::new(CpuBuffer::new_with_data(data, usage))))
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let mut buf = buffer.lock().unwrap();
        let offset = offset as usize;
        buf.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let mut bytes = vec![];
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        let mut buf = buffer.lock().unwrap();
        let offset_bytes = offset as usize * (bytes.len() / data.len());

        unsafe {
            let dst_ptr = buf.data.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst_ptr.add(offset_bytes), bytes.len());
        }

        Ok(())
    }

    async fn read_buffer<T: DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        let buf = buffer.lock().unwrap();
        let len = data.len().min(buf.len());
        data[..len].copy_from_slice(&buf.data[..len]);
        Ok(())
    }

    async fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        let buf = buffer.lock().unwrap();
        let size = buf.len() * std::mem::size_of::<T>();
        let bytes = unsafe { std::slice::from_raw_parts(buf.data.as_ptr() as *const u8, size) };

        let encase_buffer = StorageBuffer::new(bytes);
        let mut result = vec![];
        encase_buffer.read(&mut result).unwrap();
        let len = result.len().min(data.len());
        data[..len].copy_from_slice(&result[..len]);

        Ok(())
    }

    async fn slow_read_buffer<T: DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        self.read_buffer(buffer, data).await
    }
}

impl Encoder<Cpu> for CpuEncoder {
    fn begin_pass(&mut self, _label: &str, _timestamps: Option<&mut super::GpuTimestamps>) -> CpuPass {
        CpuPass {
            _operations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &Arc<Mutex<CpuBuffer<T>>>,
        source_offset: usize,
        target: &mut Arc<Mutex<CpuBuffer<T>>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), CpuBackendError> {
        // Directly copy on CPU - no need to defer
        let src = source.lock().unwrap();
        let mut dst = target.lock().unwrap();

        dst.data[target_offset..target_offset + copy_len]
            .copy_from_slice(&src.data[source_offset..source_offset + copy_len]);

        Ok(())
    }

    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &Arc<Mutex<CpuBuffer<T>>>,
        source_offset: usize,
        target: &mut Arc<Mutex<CpuBuffer<T>>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), CpuBackendError> {
        let sz = T::min_size().get() as usize;
        let sz_in_t = sz.div_ceil(std::mem::size_of::<T>());
        let copy_len_in_t = copy_len * sz_in_t;
        let src_offset_in_t = source_offset * sz_in_t;
        let dst_offset_in_t = target_offset * sz_in_t;

        let src = source.lock().unwrap();
        let mut dst = target.lock().unwrap();

        dst.data[dst_offset_in_t..dst_offset_in_t + copy_len_in_t]
            .copy_from_slice(&src.data[src_offset_in_t..src_offset_in_t + copy_len_in_t]);

        Ok(())
    }
}

impl<'a> Dispatch<'a, Cpu> for CpuDispatch<'a> {
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, Cpu>>,
        workgroups: [u32; 3],
    ) -> Result<(), CpuBackendError> {
        // Get the grid dimensions
        let grid_dim = match grid.into() {
            DispatchGrid::Direct(dims) => dims,
            DispatchGrid::Indirect(_) => {
                return Err(CpuBackendError::Cpu(
                    "Indirect dispatch not supported on CPU backend".to_string(),
                ));
            }
        };

        unsafe {
            // Calculate total thread count
            let total_x = grid_dim[0] * workgroups[0];
            let total_y = grid_dim[1] * workgroups[1];
            let total_z = grid_dim[2] * workgroups[2];
            let total_threads = (total_x * total_y * total_z) as usize;

            if total_threads == 0 {
                return Ok(());
            }

            // Collect buffer pointers in sorted order
            let mut sorted_bindings: Vec<_> = self.bindings.iter().collect();
            sorted_bindings.sort_by_key(|(k, _)| *k);

            // Extract raw buffer pointers - wrap in Send+Sync wrapper
            let buffer_ptrs: Arc<Vec<SendPtr>> = Arc::new(
                sorted_bindings
                    .iter()
                    .map(|(_, buf)| SendPtr(buf.as_ptr()))
                    .collect(),
            );

            // Try multiple entry point names for Slang's various conventions
            let entry_point_variants = vec![
                self.function.entry_point.clone(),
                format!("{}_Thread", self.function.entry_point),
                format!("{}_0", self.function.entry_point),
            ];

            let mut loaded_func = None;
            for variant in &entry_point_variants {
                let entry_cstr = std::ffi::CString::new(variant.as_str())
                    .map_err(|e| CpuBackendError::Cpu(format!("Invalid entry point: {}", e)))?;

                if let Ok(func) = self
                    .function
                    .library
                    .get::<unsafe extern "C" fn()>(entry_cstr.as_bytes())
                {
                    loaded_func = Some(func);
                    break;
                }
            }

            let loaded_func = loaded_func.ok_or_else(|| {
                CpuBackendError::Cpu(format!(
                    "Failed to load function '{}' (tried variants: {:?})",
                    self.function.entry_point, entry_point_variants
                ))
            })?;

            // Execute kernel in parallel using Rayon
            // We'll dispatch work groups in parallel
            let num_groups_x = grid_dim[0] as usize;
            let num_groups_y = grid_dim[1] as usize;
            let num_groups_z = grid_dim[2] as usize;
            let total_groups = num_groups_x * num_groups_y * num_groups_z;

            // Clone Arc for the closure
            let buffer_ptrs_clone = buffer_ptrs.clone();
            let loaded_func_clone = loaded_func.clone();

            (0..total_groups).into_par_iter().try_for_each(move |group_idx| {
                let buffer_ptrs = &buffer_ptrs_clone;
                // Calculate group IDs
                let group_z = group_idx / (num_groups_x * num_groups_y);
                let group_y = (group_idx / num_groups_x) % num_groups_y;
                let group_x = group_idx % num_groups_x;

                // Execute all threads in this work group
                for local_z in 0..workgroups[2] {
                    for local_y in 0..workgroups[1] {
                        for local_x in 0..workgroups[0] {
                            // Calculate global thread ID
                            let thread_x = group_x as u32 * workgroups[0] + local_x;
                            let thread_y = group_y as u32 * workgroups[1] + local_y;
                            let thread_z = group_z as u32 * workgroups[2] + local_z;

                            // Create ComputeVaryingInput for this thread
                            let varying_input = ComputeVaryingInput {
                                start_thread_id: [thread_x, thread_y, thread_z],
                                end_thread_id: [thread_x + 1, thread_y + 1, thread_z + 1],
                            };

                            // Call the kernel function with proper parameters
                            // Slang's host-callable signature is typically:
                            // void entry(ComputeVaryingInput* varyingInput, <buffer params>...)

                            match buffer_ptrs.len() {
                                0 => {
                                    // No parameters
                                    type KernelFn0 = unsafe extern "C" fn(*const ComputeVaryingInput);
                                    let func: libloading::Symbol<KernelFn0> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input);
                                }
                                1 => {
                                    type KernelFn1 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn1> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0);
                                }
                                2 => {
                                    type KernelFn2 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn2> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0);
                                }
                                3 => {
                                    type KernelFn3 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn3> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0);
                                }
                                4 => {
                                    type KernelFn4 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn4> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0, buffer_ptrs[3].0);
                                }
                                5 => {
                                    type KernelFn5 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn5> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0, buffer_ptrs[3].0, buffer_ptrs[4].0);
                                }
                                6 => {
                                    type KernelFn6 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn6> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0, buffer_ptrs[3].0, buffer_ptrs[4].0, buffer_ptrs[5].0);
                                }
                                7 => {
                                    type KernelFn7 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn7> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0, buffer_ptrs[3].0, buffer_ptrs[4].0, buffer_ptrs[5].0, buffer_ptrs[6].0);
                                }
                                8 => {
                                    type KernelFn8 = unsafe extern "C" fn(
                                        *const ComputeVaryingInput,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                        *mut std::ffi::c_void,
                                    );
                                    let func: libloading::Symbol<KernelFn8> =
                                        std::mem::transmute(loaded_func_clone.clone());
                                    func(&varying_input, buffer_ptrs[0].0, buffer_ptrs[1].0, buffer_ptrs[2].0, buffer_ptrs[3].0, buffer_ptrs[4].0, buffer_ptrs[5].0, buffer_ptrs[6].0, buffer_ptrs[7].0);
                                }
                                n => {
                                    return Err(CpuBackendError::Cpu(format!(
                                        "Too many buffer parameters ({}). Maximum supported is 8.",
                                        n
                                    )));
                                }
                            }
                        }
                    }
                }

                Ok::<(), CpuBackendError>(())
            })?;
        }

        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Cpu> for Arc<Mutex<CpuBuffer<T>>> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut CpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        let key = binding.space * 1000 + binding.index;
        dispatch
            .bindings
            .insert(key, Arc::new(self.clone()) as Arc<dyn BufferPointer>);
        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Cpu> for CpuBufferSlice<T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut CpuDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        let key = binding.space * 1000 + binding.index;
        dispatch
            .bindings
            .insert(key, Arc::new(self.buffer.clone()) as Arc<dyn BufferPointer>);
        Ok(())
    }
}

impl<T: DeviceValue> crate::backend::Buffer<Cpu, T> for Arc<Mutex<CpuBuffer<T>>> {
    fn is_empty(&self) -> bool {
        self.lock().unwrap().is_empty()
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.lock().unwrap().len()
    }

    fn len_encased(&self) -> usize
    where
        T: EncaseType,
    {
        let buf = self.lock().unwrap();
        (buf.len() * std::mem::size_of::<T>()) / T::SHADER_SIZE.get() as usize
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> CpuBufferSlice<T> {
        let buf = self.lock().unwrap();
        let len = buf.len();

        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => e + 1,
            std::ops::Bound::Excluded(&e) => e,
            std::ops::Bound::Unbounded => len,
        };

        CpuBufferSlice {
            buffer: Arc::new(Mutex::new(CpuBuffer {
                data: buf.data[start..end].to_vec(),
                usage: buf.usage,
            })),
            start: 0,
            end: end - start,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.lock().unwrap().usage
    }
}

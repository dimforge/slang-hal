use super::BufferUsages;
use crate::ShaderArgs;
use crate::backend::{
    Backend, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, MaybeSendSync, ShaderBinding,
};
use crate::shader::ShaderArgsError;
use bytemuck::{AnyBitPattern, NoUninit};
use encase::{ShaderType, StorageBuffer};
use metal::*;
use minislang::shader_slang;
use std::ops::RangeBounds;
use std::sync::Arc;

/// Metal backend using the metal crate.
pub struct Metal {
    device: Device,
    command_queue: CommandQueue,
}

impl Metal {
    /// Creates a new Metal backend instance.
    pub fn new() -> anyhow::Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }
}

/// Metal buffer wrapper.
pub struct MetalBuffer<T: DeviceValue> {
    buffer: Buffer,
    len: usize,
    usage: BufferUsages,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceValue> MetalBuffer<T> {
    fn new(device: &Device, len: usize, usage: BufferUsages) -> Self {
        let size = (std::mem::size_of::<T>() * len) as u64;
        let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);

        Self {
            buffer,
            len,
            usage,
            _phantom: std::marker::PhantomData,
        }
    }

    fn new_with_data(device: &Device, data: &[T], usage: BufferUsages) -> Self
    where
        T: NoUninit,
    {
        let size = (std::mem::size_of::<T>() * data.len()) as u64;
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            buffer,
            len: data.len(),
            usage,
            _phantom: std::marker::PhantomData,
        }
    }

    fn new_with_bytes(device: &Device, bytes: &[u8], len: usize, usage: BufferUsages) -> Self {
        let buffer = device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            buffer,
            len,
            usage,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn size(&self) -> u64 {
        self.buffer.length()
    }
}

/// Metal buffer slice.
#[derive(Clone)]
pub struct MetalBufferSlice {
    buffer: Buffer,
    _offset: u64,
    length: u64,
}

/// Metal command encoder.
pub struct MetalEncoder {
    command_buffer: CommandBuffer,
}

impl MetalEncoder {
    fn new(command_queue: &CommandQueue) -> Self {
        let command_buffer = command_queue.new_command_buffer().to_owned();

        Self { command_buffer }
    }

    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }

    fn finish(self) -> CommandBuffer {
        self.command_buffer
    }
}

/// Metal compute pass.
pub struct MetalPass {
    command_buffer: CommandBuffer,
}

/// Metal compute pipeline.
pub struct MetalPipeline {
    pipeline_state: ComputePipelineState,
}

/// Metal dispatch state.
pub struct MetalDispatch<'a> {
    encoder: ComputeCommandEncoder,
    pipeline: &'a MetalPipeline,
    bindings: Vec<(ShaderBinding, Buffer, u64)>,
}

#[derive(thiserror::Error, Debug)]
pub enum MetalBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error("Metal error: {0}")]
    Metal(String),
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl Backend for Metal {
    const NAME: &'static str = "metal";
    const TARGET: super::CompileTarget = super::CompileTarget::Metal;

    type Error = MetalBackendError;
    type Buffer<T: DeviceValue> = Arc<MetalBuffer<T>>;
    type BufferSlice<'b, T: DeviceValue> = MetalBufferSlice;
    type Encoder = MetalEncoder;
    type Pass = MetalPass;
    type Module = Library;
    type Function = MetalPipeline;
    type Dispatch<'a> = MetalDispatch<'a>;

    fn as_metal(&self) -> Option<&Metal> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        let src = str::from_utf8(bytes).unwrap();
        let library = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .unwrap();
        Ok(library)
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error> {
        let function = module.get_function(entry_point, None).map_err(|e| {
            MetalBackendError::Metal(format!("Function '{}' not found: {}", entry_point, e))
        })?;

        let pipeline_state = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalBackendError::Metal(e.to_string()))?;

        Ok(MetalPipeline { pipeline_state })
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        MetalEncoder::new(&self.command_queue)
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        let encoder = pass.command_buffer.new_compute_command_encoder().to_owned();

        MetalDispatch {
            encoder,
            pipeline: function,
            bindings: Vec::new(),
        }
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        let command_buffer = encoder.finish();
        command_buffer.commit();
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // Metal doesn't have a global synchronize, so we do nothing here
        // Synchronization happens per command buffer via wait_until_completed
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
        Ok(Arc::new(MetalBuffer::new_with_data(
            &self.device,
            data,
            usage,
        )))
    }

    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let mut bytes = vec![];
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        Ok(Arc::new(MetalBuffer::new_with_bytes(
            &self.device,
            &bytes,
            data.len(),
            usage,
        )))
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(Arc::new(MetalBuffer::new(&self.device, len, usage)))
    }

    fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let size = T::min_size().get() as usize * len;
        let size_in_t = size.div_ceil(std::mem::size_of::<T>());
        Ok(Arc::new(MetalBuffer::new(&self.device, size_in_t, usage)))
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let ptr = buffer.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
        }
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
        let elt_sz = bytes.len() / data.len();
        let offset_bytes = offset as usize * elt_sz;

        let ptr = buffer.buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset_bytes), bytes.len());
        }
        Ok(())
    }

    fn read_buffer<T: DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            let ptr = buffer.buffer.contents() as *const T;
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len().min(buffer.len));
            }
            Ok(())
        }
    }

    fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            let ptr = buffer.buffer.contents() as *const u8;
            let size = buffer.size() as usize;
            let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };

            let encase_buffer = StorageBuffer::new(bytes);
            let mut result = vec![];
            encase_buffer.read(&mut result).unwrap();
            let len = result.len().min(data.len());
            data[..len].copy_from_slice(&result[..len]);

            Ok(())
        }
    }

    fn slow_read_buffer<T: DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move { self.read_buffer(buffer, data).await }
    }
}

impl Encoder<Metal> for MetalEncoder {
    fn begin_pass(&mut self) -> MetalPass {
        MetalPass {
            command_buffer: self.command_buffer.to_owned(),
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &Arc<MetalBuffer<T>>,
        source_offset: usize,
        target: &mut Arc<MetalBuffer<T>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), MetalBackendError> {
        let blit_encoder = self.command_buffer.new_blit_command_encoder();
        let size = (copy_len * std::mem::size_of::<T>()) as u64;
        let src_offset = (source_offset * std::mem::size_of::<T>()) as u64;
        let dst_offset = (target_offset * std::mem::size_of::<T>()) as u64;

        blit_encoder.copy_from_buffer(&source.buffer, src_offset, &target.buffer, dst_offset, size);
        blit_encoder.end_encoding();

        Ok(())
    }

    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &Arc<MetalBuffer<T>>,
        source_offset: usize,
        target: &mut Arc<MetalBuffer<T>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), MetalBackendError> {
        let blit_encoder = self.command_buffer.new_blit_command_encoder();
        let sz = T::min_size().get() as usize;
        let size = (copy_len * sz) as u64;
        let src_offset = (source_offset * sz) as u64;
        let dst_offset = (target_offset * sz) as u64;

        blit_encoder.copy_from_buffer(&source.buffer, src_offset, &target.buffer, dst_offset, size);
        blit_encoder.end_encoding();

        Ok(())
    }
}

impl<'a> Dispatch<'a, Metal> for MetalDispatch<'a> {
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, Metal>>,
        workgroups: [u32; 3],
    ) -> Result<(), MetalBackendError> {
        self.encoder
            .set_compute_pipeline_state(&self.pipeline.pipeline_state);

        // Bind buffers
        for (i, (_binding, buffer, _size)) in self.bindings.iter().enumerate() {
            self.encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        match grid.into() {
            DispatchGrid::Direct(grid_dim) => {
                if grid_dim[0] * grid_dim[1] * grid_dim[2] > 0 {
                    let grid_size = MTLSize {
                        width: grid_dim[0] as u64,
                        height: grid_dim[1] as u64,
                        depth: grid_dim[2] as u64,
                    };
                    let threadgroup_size = MTLSize {
                        width: workgroups[0] as u64,
                        height: workgroups[1] as u64,
                        depth: workgroups[2] as u64,
                    };
                    self.encoder
                        .dispatch_thread_groups(grid_size, threadgroup_size);
                }
            }
            DispatchGrid::Indirect(grid_indirect) => {
                let threadgroup_size = MTLSize {
                    width: workgroups[0] as u64,
                    height: workgroups[1] as u64,
                    depth: workgroups[2] as u64,
                };
                self.encoder.dispatch_thread_groups_indirect(
                    &grid_indirect.buffer,
                    0,
                    threadgroup_size,
                );
            }
        }

        self.encoder.end_encoding();
        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Metal> for Arc<MetalBuffer<T>> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut MetalDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch
            .bindings
            .push((binding, self.buffer.to_owned(), self.size()));
        Ok(())
    }
}

impl<'b> ShaderArgs<'b, Metal> for MetalBufferSlice {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut MetalDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch
            .bindings
            .push((binding, self.buffer.to_owned(), self.length));
        Ok(())
    }
}

impl<T: DeviceValue> crate::backend::Buffer<Metal, T> for Arc<MetalBuffer<T>> {
    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.len
    }

    fn len_encased(&self) -> usize
    where
        T: EncaseType,
    {
        self.size() as usize / T::SHADER_SIZE.get() as usize
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> MetalBufferSlice {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s as u64 * std::mem::size_of::<T>() as u64,
            std::ops::Bound::Excluded(&s) => (s + 1) as u64 * std::mem::size_of::<T>() as u64,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => (e + 1) as u64 * std::mem::size_of::<T>() as u64,
            std::ops::Bound::Excluded(&e) => e as u64 * std::mem::size_of::<T>() as u64,
            std::ops::Bound::Unbounded => self.size(),
        };

        MetalBufferSlice {
            buffer: self.buffer.to_owned(),
            _offset: start,
            length: end - start,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.usage
    }
}

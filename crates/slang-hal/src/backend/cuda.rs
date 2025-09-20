use crate::ShaderArgs;
use crate::backend::{
    Backend, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, ShaderBinding,
};
use crate::shader::ShaderArgsError;
use bytemuck::Pod;
use cudarc::driver::safe::{CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchArgs};
use cudarc::driver::{CudaContext, CudaModule, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use minislang::shader_slang;
use std::ffi::{CStr, FromBytesWithNulError};
use std::ops::RangeBounds;
use std::sync::Arc;
use wgpu::{Buffer, BufferSlice, BufferUsages};

#[cfg(feature = "cublas")]
use cudarc::cublas::safe::CudaBlas;
use encase::ShaderType;
use encase::private::RuntimeSizedArray;

#[derive(Clone)]
pub struct Cuda {
    pub ctxt: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    #[cfg(feature = "cublas")]
    pub cublas: Arc<CudaBlas>,
    // TODO: add a more comprehensive bitmask for enabled features?
    #[cfg(feature = "cublas")]
    pub cublas_enabled: bool,
}

impl Cuda {
    pub fn new() -> Result<Self, CudaBackendError> {
        let ctxt = CudaContext::new(0)?;
        let stream = ctxt.default_stream();
        #[cfg(feature = "cublas")]
        let cublas = Arc::new(CudaBlas::new(stream.clone())?);
        Ok(Self {
            ctxt,
            stream,
            #[cfg(feature = "cublas")]
            cublas,
            #[cfg(feature = "cublas")]
            cublas_enabled: cfg!(feature = "cublas"),
        })
    }
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct ForceDeviceRepr<T: DeviceValue>(pub T);

#[derive(thiserror::Error, Debug)]
pub enum CudaBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error(transparent)]
    CudaDriver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error(transparent)]
    PtxRead(#[from] FromBytesWithNulError),
    #[cfg(feature = "cublas")]
    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),
}

unsafe impl<T: DeviceValue> DeviceRepr for ForceDeviceRepr<T> {}

#[async_trait::async_trait]
impl Backend for Cuda {
    const NAME: &'static str = "cuda";
    const TARGET: shader_slang::CompileTarget = shader_slang::CompileTarget::Ptx;

    type Error = CudaBackendError;
    type Buffer<T: DeviceValue> = CudaSlice<ForceDeviceRepr<T>>;
    type BufferSlice<'b, T: DeviceValue> = CudaView<'b, ForceDeviceRepr<T>>;
    type Encoder = Cuda;
    type Function = CudaFunction;
    type Pass = Cuda;
    type Module = Arc<CudaModule>;
    type Dispatch<'a> = LaunchArgs<'a>;

    fn as_cuda(&self) -> Option<&Cuda> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        let c_str = CStr::from_bytes_with_nul(bytes)?.to_string_lossy();
        Ok(self.ctxt.load_module(Ptx::from_src(c_str))?)
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error> {
        Ok(module.load_function(entry_point)?)
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        self.clone()
    }

    fn begin_dispatch<'a>(
        &'a self,
        _pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        self.stream.launch_builder(function)
    }

    fn submit(&self, _encoder: Self::Encoder) -> Result<(), Self::Error> {
        // No real action to perform here?
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        // TODO: doesn’t sound like the best place to set this flag.
        self.ctxt.set_blocking_synchronize()?;
        Ok(self.stream.synchronize()?)
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + Pod>(
        &self,
        data: &[T],
        _usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let wrapped: &[ForceDeviceRepr<T>] = bytemuck::try_cast_slice(data)?;
        Ok(self.stream.memcpy_stod(wrapped)?)
    }

    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        // let wrapped: &[ForceDeviceRepr<T>] = bytemuck::try_cast_slice(data)?;
        // Ok(self.stream.memcpy_stod(wrapped)?)
        todo!()
    }

    unsafe fn uninit_buffer<T: DeviceValue + Pod>(
        &self,
        len: usize,
        _usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        Ok(self.stream.alloc(len)?)
    }

    unsafe fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        _usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        // Ok(self.stream.alloc(len)?)
        todo!()
    }

    fn write_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let wrapped: &[ForceDeviceRepr<T>] = bytemuck::try_cast_slice(data)?;
        Ok(self.stream.memcpy_htod(wrapped, buffer)?)
    }

    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error> {
        // let wrapped: &[ForceDeviceRepr<T>] = bytemuck::try_cast_slice(data)?;
        // Ok(self.stream.memcpy_htod(wrapped, buffer)?)
        todo!()
    }

    async fn read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        let wrapped: &mut [ForceDeviceRepr<T>] = bytemuck::try_cast_slice_mut(data)?;
        Ok(self
            .stream
            .memcpy_dtoh(buffer, &mut wrapped[..buffer.len()])?)
    }

    async fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        // let wrapped: &mut [ForceDeviceRepr<T>] = bytemuck::try_cast_slice_mut(data)?;
        // Ok(self
        //     .stream
        //     .memcpy_dtoh(buffer, &mut wrapped[..buffer.len()])?)
        todo!()
    }

    async fn slow_read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        self.read_buffer(buffer, data).await
    }
}

impl Encoder<Cuda> for Cuda {
    fn begin_pass(&mut self) -> <Self as Backend>::Pass {
        self.clone()
    }

    fn copy_buffer_to_buffer<T: DeviceValue + Pod>(
        &mut self,
        source: &<Cuda as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <Cuda as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), <Cuda as Backend>::Error> {
        Ok(self.stream.memcpy_dtod(
            &source.slice(source_offset..source_offset + copy_len),
            &mut target.slice_mut(target_offset..target_offset + copy_len),
        )?)
    }

    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &<Cuda as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <Cuda as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), <Cuda as Backend>::Error> {
        // Ok(self.stream.memcpy_dtod(
        //     &source.slice(source_offset..source_offset + copy_len),
        //     &mut target.slice_mut(target_offset..target_offset + copy_len),
        // )?)
        todo!()
    }
}

impl<'a> Dispatch<'a, Cuda> for LaunchArgs<'a> {
    fn launch<'b>(
        mut self,
        grid: impl Into<DispatchGrid<'b, Cuda>>,
        block_dim: [u32; 3],
    ) -> Result<(), CudaBackendError> {
        match grid.into() {
            DispatchGrid::Direct(grid_dim) => {
                let config = LaunchConfig {
                    grid_dim: (grid_dim[0], grid_dim[1], grid_dim[2]),
                    block_dim: (block_dim[0], block_dim[1], block_dim[2]),
                    shared_mem_bytes: 0,
                };

                // TODO: safety?
                unsafe {
                    LaunchArgs::launch(&mut self, config)?;
                }
            }
            DispatchGrid::Indirect(grid_indirect) => {
                todo!("Indirect dispatch needs to be emulated on cuda.")
            }
        }
        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Cuda> for CudaSlice<ForceDeviceRepr<T>> {
    #[inline]
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut <Cuda as Backend>::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch.arg(self);
        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Cuda> for CudaView<'_, T> {
    #[inline]
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut <Cuda as Backend>::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch.arg(&*self);
        Ok(())
    }
}

impl<T: DeviceValue> crate::backend::Buffer<Cuda, T> for CudaSlice<ForceDeviceRepr<T>> {
    fn len(&self) -> usize {
        (*self).len()
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> <Cuda as Backend>::BufferSlice<'_, T> {
        self.slice(range)
    }
}

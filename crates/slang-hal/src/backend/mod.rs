use crate::ShaderArgs;
use crate::shader::ShaderArgsError;
use bytemuck::Pod;
use encase::internal::{CreateFrom, WriteInto};
use encase::private::ReadFrom;
use encase::{ShaderSize, ShaderType};
use minislang::shader_slang::CompileTarget;
use std::error::Error;
use std::ops::RangeBounds;
use wgpu::BufferUsages;

#[cfg(feature = "cuda")]
pub use cuda::Cuda;
pub use webgpu::WebGpu;

#[cfg(feature = "cuda")]
mod cuda;
mod webgpu;

// TODO: define our own buffer usages if we want to make wgpu optional.
pub type BufferOptions = wgpu::BufferUsages;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ShaderBinding {
    /// Binding space (aka. binding group).
    pub space: u32,
    /// Binding index.
    pub index: u32,
}

/// A value that can be sent to the GPU.
///
/// # Safety
///
/// The value must comply to the safety requirements of all the backends it is implemented for.
pub unsafe trait DeviceValue: 'static + Clone + Copy + Send + Sync {}

pub trait EncaseType: ShaderType + ShaderSize + WriteInto + CreateFrom + ReadFrom {}
impl<T: ShaderType + ShaderSize + WriteInto + CreateFrom + ReadFrom> EncaseType for T {}

// TODO: don’t do a blanket impl?
unsafe impl<T: 'static + Clone + Copy + Send + Sync> DeviceValue for T {}

#[async_trait::async_trait]
pub trait Backend: 'static + Sized + Send + Sync {
    const NAME: &'static str;
    const TARGET: CompileTarget;

    type Error: Error + Send + Sync + 'static + From<ShaderArgsError>;
    type Buffer<T: DeviceValue>: Buffer<Self, T>;
    type BufferSlice<'b, T: DeviceValue>: Send + Sync + for<'c> ShaderArgs<'c, Self>;
    type Encoder: Encoder<Self> + Send + Sync;
    type Pass: Send + Sync;
    type Module;
    type Function: Send + Sync;
    type Dispatch<'a>: Dispatch<'a, Self>
    where
        Self: 'a;

    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&crate::cuda::Cuda> {
        None
    }
    fn as_webgpu(&self) -> Option<&WebGpu> {
        None
    }

    /*
     * Module/function loading.
     */
    fn load_module(&self, data: &str) -> Result<Self::Module, Self::Error> {
        self.load_module_bytes(data.as_bytes())
    }
    fn load_module_bytes(&self, data: &[u8]) -> Result<Self::Module, Self::Error>;
    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error>;

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder;
    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a>;
    fn synchronize(&self) -> Result<(), Self::Error>;
    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error>;

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + Pod>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;
    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;

    // fn init_buffer_bytes<T: Copy>(&self, bytes: &[u8], usage: BufferUsages) -> Result<Self::Buffer<T>, Self::Error>;

    /// # Safety
    /// The returned buffer must be initialized before being read from.
    unsafe fn uninit_buffer<T: DeviceValue + Pod>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;

    /// # Safety
    /// The returned buffer must be initialized before being read from.
    unsafe fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;
    fn write_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error>;
    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error>;
    async fn read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error>;
    async fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error>;
    /// Slower version of `read_buffer` that doesn’t require `buffer` to be a mapped staging
    /// buffer.
    ///
    /// This is slower, but more convenient than [`Self::read_buffer`] because it takes care of
    /// creating a staging buffer, running a buffer-to-buffer copy from `buffer` to the staging
    /// buffer, and running a buffer-to-host copy from the staging buffer to `data`.
    async fn slow_read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error>;

    async fn slow_read_vec<T: DeviceValue + Pod + Default>(
        &self,
        buffer: &Self::Buffer<T>,
    ) -> Result<Vec<T>, Self::Error> {
        let mut result = vec![T::default(); buffer.len()];
        self.slow_read_buffer(buffer, &mut result).await?;
        Ok(result)
    }
}

pub trait Encoder<B: Backend> {
    fn begin_pass(&mut self) -> B::Pass;
    fn copy_buffer_to_buffer<T: DeviceValue + Pod>(
        &mut self,
        source: &B::Buffer<T>,
        source_offset: usize,
        target: &mut B::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), B::Error>;
    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &B::Buffer<T>,
        source_offset: usize,
        target: &mut B::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), B::Error>;
}

pub trait Dispatch<'a, B: Backend> {
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, B>>,
        workgroups: [u32; 3],
    ) -> Result<(), B::Error>;
}

pub trait Buffer<B: Backend, T: DeviceValue>: Send + Sync + for<'b> ShaderArgs<'b, B> {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize;
    fn as_slice(&self) -> B::BufferSlice<'_, T> {
        self.slice(..)
    }
    fn slice(&self, range: impl RangeBounds<usize>) -> B::BufferSlice<'_, T>;
}

pub enum DispatchGrid<'a, B: Backend> {
    Direct([u32; 3]),
    Indirect(&'a B::Buffer<[u32; 3]>),
}

impl<'a, B: Backend> From<u32> for DispatchGrid<'a, B> {
    fn from(grid: u32) -> DispatchGrid<'a, B> {
        DispatchGrid::Direct([grid, 1, 1])
    }
}

impl<'a, B: Backend> From<[u32; 3]> for DispatchGrid<'a, B> {
    fn from(grid: [u32; 3]) -> DispatchGrid<'a, B> {
        DispatchGrid::Direct(grid)
    }
}

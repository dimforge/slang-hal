use crate::ShaderArgs;
use crate::shader::ShaderArgsError;
use bytemuck::{AnyBitPattern, NoUninit};
use encase::internal::{CreateFrom, WriteInto};
use encase::private::ReadFrom;
use encase::{ShaderSize, ShaderType};
use std::error::Error;
use std::ops::RangeBounds;
use std::any::Any;

/// Shader compilation target for different backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompileTarget {
    /// WebGPU WGSL shader language
    Wgsl,
    /// Metal shading language
    Metal,
    /// Vulkan SPIR-V
    Spirv,
    /// CUDA PTX
    Ptx,
    /// Host-callable CPU code
    HostHostCallable,
}

#[cfg(feature = "runtime")]
impl From<CompileTarget> for minislang::shader_slang::CompileTarget {
    fn from(target: CompileTarget) -> Self {
        match target {
            CompileTarget::Wgsl => minislang::shader_slang::CompileTarget::Wgsl,
            CompileTarget::Metal => minislang::shader_slang::CompileTarget::Metal,
            CompileTarget::Spirv => minislang::shader_slang::CompileTarget::Spirv,
            CompileTarget::Ptx => minislang::shader_slang::CompileTarget::Ptx,
            CompileTarget::HostHostCallable => minislang::shader_slang::CompileTarget::HostHostCallable,
        }
    }
}

#[cfg(feature = "webgpu")]
pub use webgpu::WebGpu;
#[cfg(feature = "cuda")]
pub use cuda::Cuda;
#[cfg(feature = "vulkan")]
pub use vulkan::Vulkan;
#[cfg(feature = "metal")]
pub use metal::Metal;
#[cfg(feature = "cpu")]
pub use cpu::Cpu;

#[cfg(feature = "webgpu")]
mod webgpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "cpu")]
mod cpu;

bitflags::bitflags! {
    /// Buffer usage flags that mirror wgpu::BufferUsages.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsages: u32 {
        const MAP_READ = 1 << 0;
        const MAP_WRITE = 1 << 1;
        const COPY_SRC = 1 << 2;
        const COPY_DST = 1 << 3;
        const INDEX = 1 << 4;
        const VERTEX = 1 << 5;
        const UNIFORM = 1 << 6;
        const STORAGE = 1 << 7;
        const INDIRECT = 1 << 8;
        const QUERY_RESOLVE = 1 << 9;
    }
}

#[cfg(feature = "webgpu")]
impl From<BufferUsages> for wgpu::BufferUsages {
    fn from(usage: BufferUsages) -> Self {
        wgpu::BufferUsages::from_bits_truncate(usage.bits())
    }
}

#[cfg(feature = "webgpu")]
impl From<wgpu::BufferUsages> for BufferUsages {
    fn from(usage: wgpu::BufferUsages) -> Self {
        BufferUsages::from_bits_truncate(usage.bits())
    }
}

pub type BufferOptions = BufferUsages;

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
pub unsafe trait DeviceValue: 'static + Clone + Copy + MaybeSendSync {}

pub trait EncaseType: ShaderType + ShaderSize + WriteInto + CreateFrom + ReadFrom {}
impl<T: ShaderType + ShaderSize + WriteInto + CreateFrom + ReadFrom> EncaseType for T {}

// TODO: don’t do a blanket impl?
unsafe impl<T: 'static + Clone + Copy + MaybeSendSync> DeviceValue for T {}

#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {
}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSendSync for T {}

#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync> MaybeSendSync for T {}

pub trait Backend: 'static + Sized + MaybeSendSync {
    const NAME: &'static str;
    const TARGET: CompileTarget;

    type Error: Error + 'static + Send + Sync + From<ShaderArgsError>;
    type Buffer<T: DeviceValue>: MaybeSendSync + Buffer<Self, T>;
    type BufferSlice<'b, T: DeviceValue>: for<'c> ShaderArgs<'c, Self>;
    type Encoder: MaybeSendSync + Encoder<Self>;
    type Pass: MaybeSendSync;
    type Module;
    type Function: MaybeSendSync;
    type Dispatch<'a>: Dispatch<'a, Self>
    where
        Self: 'a;

    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&crate::backend::Cuda> {
        None
    }
    #[cfg(feature = "webgpu")]
    fn as_webgpu(&self) -> Option<&WebGpu> {
        None
    }
    #[cfg(feature = "vulkan")]
    fn as_vulkan(&self) -> Option<&Vulkan> {
        None
    }
    #[cfg(feature = "metal")]
    fn as_metal(&self) -> Option<&Metal> {
        None
    }
    #[cfg(feature = "cpu")]
    fn as_cpu(&self) -> Option<&Cpu> {
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
    fn init_buffer<T: DeviceValue + NoUninit>(
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

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;

    fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error>;
    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error>;
    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error>;
    fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync;
    fn read_buffer_encased<T: MaybeSendSync + DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync;
    /// Slower version of `read_buffer` that doesn’t require `buffer` to be a mapped staging
    /// buffer.
    ///
    /// This is slower, but more convenient than [`Self::read_buffer`] because it takes care of
    /// creating a staging buffer, running a buffer-to-buffer copy from `buffer` to the staging
    /// buffer, and running a buffer-to-host copy from the staging buffer to `data`.
    fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync;

    fn slow_read_vec<T: MaybeSendSync + DeviceValue + AnyBitPattern + Default>(
        &self,
        buffer: &Self::Buffer<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>> + MaybeSendSync {
        async move {
            let mut result = vec![T::default(); buffer.len()];
            self.slow_read_buffer(buffer, &mut result).await?;
            Ok(result)
        }
    }
}

pub trait Encoder<B: Backend> {
    fn begin_pass(&mut self) -> B::Pass;
    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
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

pub trait Buffer<B: Backend, T: DeviceValue>: for<'b> ShaderArgs<'b, B> {
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize
    where
        T: Sized;
    fn len_encased(&self) -> usize
    where
        T: EncaseType;
    fn as_slice(&self) -> B::BufferSlice<'_, T> {
        self.slice(..)
    }
    fn slice(&self, range: impl RangeBounds<usize>) -> B::BufferSlice<'_, T>;
    fn usage(&self) -> BufferUsages;
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

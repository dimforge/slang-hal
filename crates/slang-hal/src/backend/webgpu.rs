use crate::ShaderArgs;
use crate::backend::{
    Backend, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, ShaderBinding,
};
use crate::shader::ShaderArgsError;
use async_channel::RecvError;
use bytemuck::Pod;
use encase::{ShaderType, StorageBuffer};
use minislang::shader_slang;
use regex::Regex;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::RangeBounds;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::CommandEncoderDescriptor;
use wgpu::{
    Adapter, Buffer, BufferAddress, BufferDescriptor, BufferSlice, BufferUsages, BufferView,
    CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Instance, PipelineCompilationOptions, PollError, Queue, ShaderModule,
    ShaderRuntimeChecks,
};

/// Helper struct to initialize a device and its queue.
pub struct WebGpu {
    _instance: Instance, // TODO: do we have to keep this around?
    _adapter: Adapter,   // TODO: do we have to keep this around?
    device: Device,
    queue: Queue,
    hacks: Vec<(Regex, String)>,
    /// If this flag is set, every buffer created by this backend will have the
    /// `BufferUsages::COPY_SRC` flag. Useful for debugging.
    pub force_buffer_copy_src: bool,
}

impl WebGpu {
    pub async fn default() -> anyhow::Result<Self> {
        Self::new(wgpu::Features::default(), wgpu::Limits::default()).await
    }

    /// Initializes a wgpu instance and create its queue.
    pub async fn new(features: wgpu::Features, limits: wgpu::Limits) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to initialize gpu adapter."))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            force_buffer_copy_src: false,
            hacks: vec![],
        })
    }

    pub fn append_hack(&mut self, regex: Regex, replace_pattern: String) {
        self.hacks.push((regex, replace_pattern));
    }

    /// The `wgpu` device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The `wgpu` queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

#[derive(thiserror::Error, Debug)]
pub enum WebGpuBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error(transparent)]
    Wgpu(#[from] wgpu::Error),
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error("Failed to read buffer from GPU: {0}")]
    BufferRead(RecvError),
    #[error(transparent)]
    DevicePoll(#[from] PollError),
}

#[async_trait::async_trait]
impl Backend for WebGpu {
    const NAME: &'static str = "webgpu";
    const TARGET: shader_slang::CompileTarget = shader_slang::CompileTarget::Wgsl;

    type Error = WebGpuBackendError;
    type Buffer<T: DeviceValue> = Buffer;
    type BufferSlice<'b, T: DeviceValue> = BufferSlice<'b>;
    type Encoder = wgpu::CommandEncoder;
    type Pass = ComputePass<'static>;
    type Module = ShaderModule;
    type Function = wgpu::ComputePipeline;
    type Dispatch<'a> = WebGpuDispatch<'a>;

    fn as_webgpu(&self) -> Option<&WebGpu> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module(&self, data: &str) -> Result<Self::Module, Self::Error> {
        // HACK: slang tends to introduce some useless conversions when unpacking, resulting in
        //       the SHADER_F16 feature being needed for no good reasons.
        let mut data = data.replace("enable f16;", "").replace("f16", "f32");

        // Apply other user-defined hacks.
        for (reg, replace) in &self.hacks {
            data = reg.replace_all(&data, replace).to_string();
        }

        let module = unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&data)),
                },
                ShaderRuntimeChecks::unchecked(),
            )
        };
        Ok(module)
    }

    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        self.load_module(str::from_utf8(bytes).unwrap())
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error> {
        /*
         * Create the pipeline.
         */
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });

        Ok(pipeline)
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor::default())
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> WebGpuDispatch<'a> {
        WebGpuDispatch::new(&self.device, pass, function)
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        let _ = self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + Pod>(
        &self,
        data: &[T],
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        Ok(self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::try_cast_slice(data)?,
            usage,
        }))
    }

    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        let mut bytes = vec![]; // TODO PERF: can we avoid the allocation somehow?
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        Ok(self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage,
        }))
    }

    // fn init_buffer_bytes<T: Copy>(&self, data: &[u8], usage: BufferUsages) -> Result<Self::Buffer<T>, Self::Error> {
    //     Ok(self.device.create_buffer_init(&BufferInitDescriptor {
    //         label: None,
    //         contents: data,
    //         usage,
    //     }))
    // }

    unsafe fn uninit_buffer<T: DeviceValue + Pod>(
        &self,
        len: usize,
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        let bytes_len = std::mem::size_of::<T>() as u64 * len as u64;
        Ok(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes_len,
            usage,
            mapped_at_creation: false,
        }))
    }

    unsafe fn uninit_buffer_encased<T: DeviceValue + ShaderType>(
        &self,
        len: usize,
        mut usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        if self.force_buffer_copy_src && !usage.contains(BufferUsages::MAP_READ) {
            usage |= BufferUsages::COPY_SRC;
        }

        let bytes_len = T::min_size().get() * len as u64;
        Ok(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: bytes_len,
            usage,
            mapped_at_creation: false,
        }))
    }

    fn write_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error> {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }
    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let mut bytes = vec![]; // TODO: can we avoid the allocation?
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        self.queue.write_buffer(buffer, 0, &bytes);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        self.device.poll(wgpu::PollType::wait())?;
        Ok(())
    }

    async fn read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        let data = read_bytes(&self.device, buffer).await?;
        let result = bytemuck::try_cast_slice(&data)?;
        out[..result.len()].copy_from_slice(result);
        drop(data);
        buffer.unmap();
        Ok(())
    }

    async fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        let data = read_bytes(&self.device, buffer).await?;

        let mut result = vec![];
        let bytes = data.as_ref();
        let encase_buffer = StorageBuffer::new(&bytes);
        encase_buffer.read(&mut result).unwrap(); // TODO: propagate error
        out[..result.len()].copy_from_slice(&result);

        drop(data);
        buffer.unmap();
        Ok(())
    }

    async fn slow_read_buffer<T: DeviceValue + Pod>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> Result<(), Self::Error> {
        // Create staging buffer.
        // SAFETY: the buffer will be initialized by a buffer-to-buffer copy.
        let bytes_len = buffer.size() as usize;
        let staging = unsafe {
            // TODO: not using `u8` because it doesn’t implement ShaderType
            self.uninit_buffer::<u32>(
                bytes_len.div_ceil(4),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )?
        };
        let mut encoder = self.begin_encoding();
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes_len as u64);
        self.submit(encoder)?;

        // Read the buffer.
        Ok(self.read_buffer(&staging, out).await?)
    }
}

impl Encoder<WebGpu> for wgpu::CommandEncoder {
    fn begin_pass(&mut self) -> ComputePass<'static> {
        self.compute_pass("").forget_lifetime()
    }

    fn copy_buffer_to_buffer<T: DeviceValue + Pod>(
        &mut self,
        source: &<WebGpu as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <WebGpu as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), WebGpuBackendError> {
        wgpu::CommandEncoder::copy_buffer_to_buffer(
            self,
            source,
            source_offset as BufferAddress * size_of::<T>() as BufferAddress,
            target,
            target_offset as BufferAddress * size_of::<T>() as BufferAddress,
            copy_len as BufferAddress * size_of::<T>() as BufferAddress,
        );
        Ok(())
    }

    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &<WebGpu as Backend>::Buffer<T>,
        source_offset: usize,
        target: &mut <WebGpu as Backend>::Buffer<T>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), WebGpuBackendError> {
        let sz = T::min_size().get() as usize;
        wgpu::CommandEncoder::copy_buffer_to_buffer(
            self,
            source,
            source_offset as BufferAddress * sz as BufferAddress,
            target,
            target_offset as BufferAddress * sz as BufferAddress,
            copy_len as BufferAddress * sz as BufferAddress,
        );
        Ok(())
    }
}

impl<'a> Dispatch<'a, WebGpu> for WebGpuDispatch<'a> {
    // NOTE: the block_dim is configured in the shader…
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, WebGpu>>,
        _block_dim: [u32; 3],
    ) -> Result<(), WebGpuBackendError> {
        if !self.launchable {
            return Ok(());
        }

        self.pass.set_pipeline(&self.pipeline);

        // TODO: we could store the BindGroupEntry directly?
        let entries: SmallVec<[_; 10]> = self
            .args
            .iter()
            .map(|(id, input)| wgpu::BindGroupEntry {
                binding: id.index,
                resource: (*input).into(),
            })
            .collect();
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        self.pass.set_bind_group(0, &bind_group, &[]);

        match grid.into() {
            DispatchGrid::Direct(grid_dim) => {
                // NOTE: we don’t need to queue if the workgroup is empty.
                if grid_dim[0] * grid_dim[1] * grid_dim[2] > 0 {
                    self.pass
                        .dispatch_workgroups(grid_dim[0], grid_dim[1], grid_dim[2]);
                }
            }
            DispatchGrid::Indirect(grid_indirect) => {
                self.pass.dispatch_workgroups_indirect(grid_indirect, 0);
            }
        }

        Ok(())
    }
}

pub struct WebGpuDispatch<'a> {
    // NOTE: keep up to 10 bindings on the stack. This number was chosen to match
    //       the current (06/2025) max storage bindings on the browser.
    device: Device,
    pass: &'a mut ComputePass<'static>,
    pipeline: ComputePipeline,
    args: SmallVec<[(ShaderBinding, BufferSlice<'a>); 10]>,
    launchable: bool,
}

impl<'a> WebGpuDispatch<'a> {
    fn new(
        device: &Device,
        pass: &'a mut ComputePass<'static>,
        pipeline: &ComputePipeline,
    ) -> WebGpuDispatch<'a> {
        WebGpuDispatch {
            device: device.clone(),
            pass,
            pipeline: pipeline.clone(),
            args: SmallVec::default(),
            launchable: true,
        }
    }
}

pub trait CommandEncoderExt {
    fn compute_pass<'encoder>(
        &'encoder mut self,
        label: &str,
        // timestamps: Option<&mut GpuTimestamps>,
    ) -> ComputePass<'encoder>;
}

impl CommandEncoderExt for CommandEncoder {
    fn compute_pass<'encoder>(
        &'encoder mut self,
        label: &str,
        // timestamps: Option<&mut GpuTimestamps>,
    ) -> ComputePass<'encoder> {
        let desc = ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None, // timestamps.and_then(|ts| ts.next_compute_pass_timestamp_writes()),
        };
        self.begin_compute_pass(&desc)
    }
}

async fn read_bytes<'a>(
    device: &Device,
    buffer: &'a Buffer,
) -> Result<BufferView<'a>, WebGpuBackendError> {
    let buffer_slice = buffer.slice(..);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, receiver) = async_channel::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send_blocking(v).unwrap()
        });
        device.poll(wgpu::PollType::wait())?;
        receiver
            .recv()
            .await
            .map_err(WebGpuBackendError::BufferRead)?
            .unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (sender, receiver) = async_channel::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.force_send(v).unwrap();
        });
        device.poll(wgpu::PollType::wait());
        receiver.recv().await?.unwrap();
    }

    let data = buffer_slice.get_mapped_range();
    Ok(data)
}

impl<'b> ShaderArgs<'b, WebGpu> for Buffer {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut <WebGpu as Backend>::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch.args.push((binding, self.slice(..)));
        Ok(())
    }
}

impl<'b> ShaderArgs<'b, WebGpu> for BufferSlice<'_> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut <WebGpu as Backend>::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch.args.push((binding, *self));
        Ok(())
    }
}

impl<T: DeviceValue> crate::backend::Buffer<WebGpu, T> for Buffer {
    fn len(&self) -> usize {
        self.size() as usize / std::mem::size_of::<T>()
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> <WebGpu as Backend>::BufferSlice<'_, T> {
        let start = range
            .start_bound()
            .map(|val| *val as u64 * std::mem::size_of::<T>() as u64);
        let end = range
            .end_bound()
            .map(|val| *val as u64 * std::mem::size_of::<T>() as u64);
        self.slice((start, end))
    }
}

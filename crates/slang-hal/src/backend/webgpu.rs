#![allow(clippy::manual_async_fn)]

use crate::ShaderArgs;
use crate::backend::{
    Backend, BufferUsages, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, MaybeSendSync,
    ShaderBinding,
};
use crate::shader::ShaderArgsError;
use async_channel::RecvError;
use bytemuck::{AnyBitPattern, NoUninit};
use encase::{ShaderType, StorageBuffer};
use regex::Regex;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::ops::RangeBounds;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::wgt::CommandEncoderDescriptor;
use wgpu::{
    Adapter, Buffer, BufferAddress, BufferDescriptor, BufferSlice, BufferView, CommandEncoder,
    ComputePass, ComputePassDescriptor, ComputePassTimestampWrites, ComputePipeline,
    ComputePipelineDescriptor, Device, ExperimentalFeatures, Instance,
    PipelineCompilationOptions, PollError, QuerySet, QuerySetDescriptor, QueryType, Queue,
    ShaderModule, ShaderRuntimeChecks,
};
use std::time::Duration;

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
                experimental_features: ExperimentalFeatures::default(),
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
    // #[error(transparent)]
    // Wgpu(#[from] wgpu::Error), // Doesn’t implement Send+Sync
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error("Failed to read buffer from GPU: {0}")]
    BufferRead(RecvError),
    #[error(transparent)]
    DevicePoll(#[from] PollError),
    #[error(transparent)]
    Recv(#[from] RecvError),
}

impl Backend for WebGpu {
    const NAME: &'static str = "webgpu";
    const TARGET: super::CompileTarget = super::CompileTarget::Wgsl;

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
    fn init_buffer<T: DeviceValue + NoUninit>(
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
            usage: usage.into(),
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
            usage: usage.into(),
        }))
    }

    // fn init_buffer_bytes<T: Copy>(&self, data: &[u8], usage: BufferUsages) -> Result<Self::Buffer<T>, Self::Error> {
    //     Ok(self.device.create_buffer_init(&BufferInitDescriptor {
    //         label: None,
    //         contents: data,
    //         usage,
    //     }))
    // }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
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
            usage: usage.into(),
            mapped_at_creation: false,
        }))
    }

    fn uninit_buffer_encased<T: DeviceValue + ShaderType>(
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
            usage: usage.into(),
            mapped_at_creation: false,
        }))
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let elt_sz = std::mem::size_of::<T>() as u64;
        self.queue
            .write_buffer(buffer, offset * elt_sz, bytemuck::cast_slice(data));
        Ok(())
    }
    fn write_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let mut bytes = vec![]; // TODO: can we avoid the allocation?
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();
        let elt_sz = bytes.len() / data.len();

        self.queue
            .write_buffer(buffer, offset * elt_sz as u64, &bytes);
        Ok(())
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    fn read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            let data = read_bytes(&self.device, buffer).await?;
            let result = bytemuck::try_cast_slice(&data)?;
            let to_copy = result.len().min(out.len());
            out[..to_copy].copy_from_slice(&result[..to_copy]);
            drop(data);
            buffer.unmap();
            Ok(())
        }
    }

    fn read_buffer_encased<T: MaybeSendSync + DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            let data = read_bytes(&self.device, buffer).await?;

            let mut result = vec![];
            let bytes = data.as_ref();
            let encase_buffer = StorageBuffer::new(&bytes);
            encase_buffer.read(&mut result).unwrap(); // TODO: propagate error
            let to_copy = result.len().min(out.len());
            out[..to_copy].copy_from_slice(&result[..to_copy]);

            drop(data);
            buffer.unmap();
            Ok(())
        }
    }

    fn slow_read_buffer<T: MaybeSendSync + DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        out: &mut [T],
    ) -> impl Future<Output = Result<(), Self::Error>> + MaybeSendSync {
        async move {
            // Create staging buffer.
            let bytes_len = buffer.size() as usize;
            // TODO: not using `u8` because it doesn’t implement ShaderType
            let staging = self.uninit_buffer::<u32>(
                bytes_len.div_ceil(4),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )?;
            let mut encoder = self.begin_encoding();
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes_len as u64);
            self.submit(encoder)?;

            // Read the buffer.
            self.read_buffer(&staging, out).await
        }
    }
}

impl Encoder<WebGpu> for wgpu::CommandEncoder {
    fn begin_pass(&mut self, label: &str, timestamps: Option<&mut GpuTimestamps>) -> ComputePass<'static> {
        if let Some(ts) = timestamps {
            let begin = ts.next_query_index;
            let end = begin + 1;
            ts.next_query_index += 2;
            ts.labels.push(label.to_string());
            let ts_writes = ComputePassTimestampWrites {
                query_set: &ts.query_set,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: Some(end),
            };
            self.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: Some(ts_writes),
            }).forget_lifetime()
        } else {
            self.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            }).forget_lifetime()
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
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

/// Result of a GPU timestamp query for a single compute pass.
#[derive(Clone, Debug)]
pub struct GpuTimingResult {
    pub label: String,
    pub duration: Duration,
}

/// GPU timestamp query context for profiling compute passes.
///
/// Create once, pass to `encoder.begin_pass(label, Some(&mut timestamps))` for each pass
/// you want to time. After recording all passes, call `resolve` before submitting the encoder,
/// then `read_results` after synchronization.
pub struct GpuTimestamps {
    query_set: QuerySet,
    resolve_buffer: Buffer,   // QUERY_RESOLVE | COPY_SRC
    staging_buffer: Buffer,   // MAP_READ | COPY_DST
    labels: Vec<String>,
    next_query_index: u32,
    capacity: u32,
    timestamp_period: f32,
}

impl GpuTimestamps {
    /// Creates a new timestamp query context.
    ///
    /// `max_passes` is the maximum number of compute passes that can be timed
    /// before `reset()` must be called.
    pub fn new(device: &Device, queue: &Queue, max_passes: u32) -> Self {
        let query_count = max_passes * 2; // begin + end per pass
        let bytes = query_count as u64 * 8; // each timestamp is u64

        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("GpuTimestamps query set"),
            count: query_count,
            ty: QueryType::Timestamp,
        });

        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("GpuTimestamps resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("GpuTimestamps staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            labels: Vec::with_capacity(max_passes as usize),
            next_query_index: 0,
            capacity: max_passes,
            timestamp_period: queue.get_timestamp_period(),
        }
    }

    /// Number of passes recorded so far.
    pub fn num_recorded_passes(&self) -> u32 {
        self.next_query_index / 2
    }

    /// Resolve timestamp queries into the staging buffer.
    /// Call this after all passes are recorded but before submitting the encoder.
    pub fn resolve(&self, encoder: &mut CommandEncoder) {
        if self.next_query_index == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.next_query_index, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            self.next_query_index as u64 * 8,
        );
    }

    /// Read timing results from the staging buffer.
    /// Call after `resolve`, `submit`, and `synchronize`.
    pub async fn read_results(&self, device: &Device) -> Result<Vec<GpuTimingResult>, WebGpuBackendError> {
        if self.next_query_index == 0 {
            return Ok(vec![]);
        }

        let data = read_bytes(device, &self.staging_buffer).await?;
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        let num_passes = self.labels.len();
        let mut results = Vec::with_capacity(num_passes);

        for i in 0..num_passes {
            let begin = timestamps[i * 2];
            let end = timestamps[i * 2 + 1];
            let nanos = (end.saturating_sub(begin)) as f64 * self.timestamp_period as f64;
            results.push(GpuTimingResult {
                label: self.labels[i].clone(),
                duration: Duration::from_nanos(nanos as u64),
            });
        }

        drop(data);
        self.staging_buffer.unmap();
        Ok(results)
    }

    /// Reset for the next frame/step.
    pub fn reset(&mut self) {
        self.next_query_index = 0;
        self.labels.clear();
    }
}

async fn read_bytes(device: &Device, buffer: &Buffer) -> Result<BufferView, WebGpuBackendError> {
    let buffer_slice = buffer.slice(..);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, receiver) = async_channel::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send_blocking(v).unwrap()
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
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
        device.poll(wgpu::PollType::wait_indefinitely())?;
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
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn len(&self) -> usize
    where
        T: Sized,
    {
        self.size() as usize / std::mem::size_of::<T>()
    }

    fn len_encased(&self) -> usize
    where
        T: EncaseType,
    {
        self.size() as usize / T::SHADER_SIZE.get() as usize
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

    fn usage(&self) -> BufferUsages {
        self.usage().into()
    }
}

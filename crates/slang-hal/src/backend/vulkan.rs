use super::BufferUsages;
use crate::ShaderArgs;
use crate::backend::{
    Backend, DeviceValue, Dispatch, DispatchGrid, EncaseType, Encoder, ShaderBinding,
};
use crate::shader::ShaderArgsError;
use ash::vk;
use bytemuck::{AnyBitPattern, NoUninit};
use encase::{ShaderType, StorageBuffer};
use minislang::shader_slang;
use std::ops::RangeBounds;
use std::sync::Arc;

/// Vulkan backend using the ash crate.
pub struct Vulkan {
    _entry: ash::Entry,
    instance: ash::Instance,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    _queue_family_index: u32,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Vulkan {
    /// Creates a new Vulkan backend instance.
    pub fn new() -> anyhow::Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;

            // Create instance
            let app_info = vk::ApplicationInfo::default()
                .application_name(c"slang-hal")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"slang-hal")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::API_VERSION_1_3);

            let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

            let instance = entry.create_instance(&create_info, None)?;

            // Pick physical device
            let physical_devices = instance.enumerate_physical_devices()?;
            let physical_device = physical_devices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No Vulkan physical devices found"))?;

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            // Find compute queue family
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family_index = queue_family_properties
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(index, _)| index as u32)
                .ok_or_else(|| anyhow::anyhow!("No compute queue family found"))?;

            // Create logical device
            let queue_priorities = [1.0];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info));

            let device = instance.create_device(physical_device, &device_create_info, None)?;
            let queue = device.get_device_queue(queue_family_index, 0);

            // Create command pool
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device.create_command_pool(&command_pool_info, None)?;

            // Create descriptor pool
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1024,
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1024)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

            let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

            Ok(Self {
                _entry: entry,
                instance,
                _physical_device: physical_device,
                device,
                queue,
                _queue_family_index: queue_family_index,
                command_pool,
                descriptor_pool,
                memory_properties,
            })
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Vulkan buffer wrapper.
pub struct VulkanBuffer<T: DeviceValue> {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    device: ash::Device,
    usage: BufferUsages,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceValue> VulkanBuffer<T> {
    fn new(
        device: &ash::Device,
        size: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        usage: BufferUsages,
    ) -> anyhow::Result<Self> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = device.create_buffer(&buffer_info, None)?;
            let mem_requirements = device.get_buffer_memory_requirements(buffer);

            let memory_type_index = Self::find_memory_type_static(
                mem_requirements.memory_type_bits,
                memory_properties,
                memory_props,
            )
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable memory type"))?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let memory = device.allocate_memory(&alloc_info, None)?;
            device.bind_buffer_memory(buffer, memory, 0)?;

            Ok(Self {
                buffer,
                memory,
                size,
                device: device.clone(),
                usage,
                _phantom: std::marker::PhantomData,
            })
        }
    }

    fn find_memory_type_static(
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
    ) -> Option<u32> {
        for i in 0..memory_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && memory_props.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Some(i);
            }
        }
        None
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}

impl<T: DeviceValue> Drop for VulkanBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Vulkan buffer slice.
pub struct VulkanBufferSlice {
    buffer: vk::Buffer,
    _offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

/// Vulkan command encoder.
pub struct VulkanEncoder {
    device: ash::Device,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

impl VulkanEncoder {
    fn new(device: &ash::Device, command_pool: vk::CommandPool) -> anyhow::Result<Self> {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers = device.allocate_command_buffers(&alloc_info)?;
            let command_buffer = command_buffers[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(command_buffer, &begin_info)?;

            Ok(Self {
                device: device.clone(),
                command_buffer,
                command_pool,
            })
        }
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    fn finish(self) -> anyhow::Result<vk::CommandBuffer> {
        unsafe {
            self.device.end_command_buffer(self.command_buffer)?;
            Ok(self.command_buffer)
        }
    }
}

impl Drop for VulkanEncoder {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
        }
    }
}

/// Vulkan compute pass (no-op for Vulkan as we don't use render passes for compute).
pub struct VulkanPass {
    encoder: VulkanEncoder,
}

/// Vulkan shader module and pipeline.
pub struct VulkanPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    device: ash::Device,
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Vulkan dispatch state.
pub struct VulkanDispatch<'a> {
    device: &'a ash::Device,
    encoder: &'a mut VulkanEncoder,
    pipeline: &'a VulkanPipeline,
    descriptor_pool: vk::DescriptorPool,
    bindings: Vec<(ShaderBinding, vk::Buffer, vk::DeviceSize)>,
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanBackendError {
    #[error(transparent)]
    ShaderArg(#[from] ShaderArgsError),
    #[error(transparent)]
    Vk(#[from] vk::Result),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
    #[error(transparent)]
    BytemuckPod(#[from] bytemuck::PodCastError),
    #[error("Failed to load Vulkan library: {0}")]
    LoadError(#[from] ash::LoadingError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[async_trait::async_trait]
impl Backend for Vulkan {
    const NAME: &'static str = "vulkan";
    const TARGET: super::CompileTarget = super::CompileTarget::Spirv;

    type Error = VulkanBackendError;
    type Buffer<T: DeviceValue> = Arc<VulkanBuffer<T>>;
    type BufferSlice<'b, T: DeviceValue> = VulkanBufferSlice;
    type Encoder = VulkanEncoder;
    type Pass = VulkanPass;
    type Module = vk::ShaderModule;
    type Function = VulkanPipeline;
    type Dispatch<'a> = VulkanDispatch<'a>;

    fn as_vulkan(&self) -> Option<&Vulkan> {
        Some(self)
    }

    /*
     * Module/function loading.
     */
    fn load_module_bytes(&self, bytes: &[u8]) -> Result<Self::Module, Self::Error> {
        unsafe {
            // SPIR-V must be 4-byte aligned
            let spirv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;

            let create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);

            let shader_module = self.device.create_shader_module(&create_info, None)?;
            Ok(shader_module)
        }
    }

    fn load_function(
        &self,
        module: &Self::Module,
        entry_point: &str,
    ) -> Result<Self::Function, Self::Error> {
        unsafe {
            // Create descriptor set layout for storage buffers
            let bindings = vec![
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            let descriptor_set_layout = self
                .device
                .create_descriptor_set_layout(&layout_info, None)?;

            // Create pipeline layout
            let layouts = [descriptor_set_layout];
            let pipeline_layout_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)?;

            // Create compute pipeline
            let entry_name = std::ffi::CString::new(entry_point).unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(*module)
                .name(&entry_name);

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(pipeline_layout);

            let pipelines = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| e.1)?;

            Ok(VulkanPipeline {
                pipeline: pipelines[0],
                pipeline_layout,
                descriptor_set_layout,
                device: self.device.clone(),
            })
        }
    }

    /*
     * Kernel dispatch.
     */
    fn begin_encoding(&self) -> Self::Encoder {
        VulkanEncoder::new(&self.device, self.command_pool).unwrap()
    }

    fn begin_dispatch<'a>(
        &'a self,
        pass: &'a mut Self::Pass,
        function: &'a Self::Function,
    ) -> Self::Dispatch<'a> {
        VulkanDispatch {
            device: &self.device,
            encoder: &mut pass.encoder,
            pipeline: function,
            descriptor_pool: self.descriptor_pool,
            bindings: Vec::new(),
        }
    }

    fn submit(&self, encoder: Self::Encoder) -> Result<(), Self::Error> {
        unsafe {
            let command_buffer = encoder.finish()?;

            let cmd_bufs = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;

            Ok(())
        }
    }

    fn synchronize(&self) -> Result<(), Self::Error> {
        unsafe {
            self.device.queue_wait_idle(self.queue)?;
            Ok(())
        }
    }

    /*
     * Buffer handling.
     */
    fn init_buffer<T: DeviceValue + NoUninit>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
        let buffer = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.memory_properties,
            usage,
        )?;

        // Create staging buffer
        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_WRITE,
        )?;

        unsafe {
            // Map and copy data to staging buffer
            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );
            self.device.unmap_memory(staging_buffer.memory);

            // Copy from staging to device buffer
            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                staging_buffer.buffer(),
                buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        Ok(Arc::new(buffer))
    }

    fn init_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        data: &[T],
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let mut bytes = vec![];
        let mut bytes_buffer = StorageBuffer::new(&mut bytes);
        bytes_buffer.write(data).unwrap();

        let size = bytes.len() as vk::DeviceSize;
        let buffer = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.memory_properties,
            usage,
        )?;

        // Create staging buffer and upload
        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_WRITE,
        )?;

        unsafe {
            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size as usize);
            self.device.unmap_memory(staging_buffer.memory);

            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                staging_buffer.buffer(),
                buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        Ok(Arc::new(buffer))
    }

    fn uninit_buffer<T: DeviceValue + NoUninit>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let size = (std::mem::size_of::<T>() * len) as vk::DeviceSize;
        let buffer = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.memory_properties,
            usage,
        )?;

        Ok(Arc::new(buffer))
    }

    fn uninit_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        len: usize,
        usage: BufferUsages,
    ) -> Result<Self::Buffer<T>, Self::Error> {
        let size = T::min_size().get() * len as u64;
        let buffer = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &self.memory_properties,
            usage,
        )?;

        Ok(Arc::new(buffer))
    }

    fn write_buffer<T: DeviceValue + NoUninit>(
        &self,
        buffer: &mut Self::Buffer<T>,
        offset: u64,
        data: &[T],
    ) -> Result<(), Self::Error> {
        let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
        let offset_bytes = offset * std::mem::size_of::<T>() as u64;

        // Create staging buffer
        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_WRITE,
        )?;

        unsafe {
            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr as *mut u8,
                size as usize,
            );
            self.device.unmap_memory(staging_buffer.memory);

            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset_bytes)
                .size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                staging_buffer.buffer(),
                buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
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
        let size = bytes.len() as vk::DeviceSize;
        let elt_sz = bytes.len() / data.len();
        let offset_bytes = offset * elt_sz as u64;

        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_WRITE,
        )?;

        unsafe {
            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size as usize);
            self.device.unmap_memory(staging_buffer.memory);

            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset_bytes)
                .size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                staging_buffer.buffer(),
                buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        Ok(())
    }

    async fn read_buffer<T: DeviceValue + AnyBitPattern>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        let size = buffer.size();

        // Create staging buffer
        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_READ,
        )?;

        unsafe {
            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                buffer.buffer(),
                staging_buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;

            // Map and read data
            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(
                ptr as *const u8,
                data.as_mut_ptr() as *mut u8,
                size as usize,
            );
            self.device.unmap_memory(staging_buffer.memory);
        }

        Ok(())
    }

    async fn read_buffer_encased<T: DeviceValue + EncaseType>(
        &self,
        buffer: &Self::Buffer<T>,
        data: &mut [T],
    ) -> Result<(), Self::Error> {
        let size = buffer.size();

        let staging_buffer: VulkanBuffer<T> = VulkanBuffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &self.memory_properties,
            BufferUsages::MAP_READ,
        )?;

        unsafe {
            let command_buffer = VulkanEncoder::new(&self.device, self.command_pool)?;
            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(
                command_buffer.command_buffer(),
                buffer.buffer(),
                staging_buffer.buffer(),
                &[region],
            );

            let cmd_buf = command_buffer.finish()?;
            let cmd_bufs = [cmd_buf];
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
            let submit_infos = [submit_info];
            self.device
                .queue_submit(self.queue, &submit_infos, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;

            let ptr = self.device.map_memory(
                staging_buffer.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            let bytes = std::slice::from_raw_parts(ptr as *const u8, size as usize);
            let encase_buffer = StorageBuffer::new(bytes);
            let mut result = vec![];
            encase_buffer.read(&mut result).unwrap();
            data[..result.len()].copy_from_slice(&result);
            self.device.unmap_memory(staging_buffer.memory);
        }

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

impl Encoder<Vulkan> for VulkanEncoder {
    fn begin_pass(&mut self, _label: &str, _timestamps: Option<&mut super::GpuTimestamps>) -> VulkanPass {
        VulkanPass {
            encoder: VulkanEncoder {
                device: self.device.clone(),
                command_buffer: self.command_buffer,
                command_pool: self.command_pool,
            },
        }
    }

    fn copy_buffer_to_buffer<T: DeviceValue + NoUninit>(
        &mut self,
        source: &Arc<VulkanBuffer<T>>,
        source_offset: usize,
        target: &mut Arc<VulkanBuffer<T>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), VulkanBackendError> {
        let size = (copy_len * std::mem::size_of::<T>()) as vk::DeviceSize;
        let src_offset = (source_offset * std::mem::size_of::<T>()) as vk::DeviceSize;
        let dst_offset = (target_offset * std::mem::size_of::<T>()) as vk::DeviceSize;

        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);
            self.device.cmd_copy_buffer(
                self.command_buffer,
                source.buffer(),
                target.buffer(),
                &[region],
            );
        }

        Ok(())
    }

    fn copy_buffer_to_buffer_encased<T: DeviceValue + ShaderType>(
        &mut self,
        source: &Arc<VulkanBuffer<T>>,
        source_offset: usize,
        target: &mut Arc<VulkanBuffer<T>>,
        target_offset: usize,
        copy_len: usize,
    ) -> Result<(), VulkanBackendError> {
        let sz = T::min_size().get() as usize;
        let size = (copy_len * sz) as vk::DeviceSize;
        let src_offset = (source_offset * sz) as vk::DeviceSize;
        let dst_offset = (target_offset * sz) as vk::DeviceSize;

        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size);
            self.device.cmd_copy_buffer(
                self.command_buffer,
                source.buffer(),
                target.buffer(),
                &[region],
            );
        }

        Ok(())
    }
}

impl<'a> Dispatch<'a, Vulkan> for VulkanDispatch<'a> {
    fn launch<'b>(
        self,
        grid: impl Into<DispatchGrid<'b, Vulkan>>,
        _workgroups: [u32; 3],
    ) -> Result<(), VulkanBackendError> {
        unsafe {
            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.pipeline.descriptor_set_layout));

            let descriptor_sets = self.device.allocate_descriptor_sets(&alloc_info)?;
            let descriptor_set = descriptor_sets[0];

            // Update descriptor sets with buffer bindings
            let mut buffer_infos = Vec::new();
            let mut write_descriptor_sets = Vec::new();

            for (_binding, buffer, size) in &self.bindings {
                let buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .offset(0)
                    .range(*size);
                buffer_infos.push(buffer_info);
            }

            for (i, (binding, _, _)) in self.bindings.iter().enumerate() {
                let write_set = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding.index)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[i]));
                write_descriptor_sets.push(write_set);
            }

            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);

            // Bind pipeline and descriptor sets
            self.device.cmd_bind_pipeline(
                self.encoder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                self.encoder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Dispatch
            match grid.into() {
                DispatchGrid::Direct(grid_dim) => {
                    if grid_dim[0] * grid_dim[1] * grid_dim[2] > 0 {
                        self.device.cmd_dispatch(
                            self.encoder.command_buffer,
                            grid_dim[0],
                            grid_dim[1],
                            grid_dim[2],
                        );
                    }
                }
                DispatchGrid::Indirect(grid_indirect) => {
                    self.device.cmd_dispatch_indirect(
                        self.encoder.command_buffer,
                        grid_indirect.buffer(),
                        0,
                    );
                }
            }
        }

        Ok(())
    }
}

impl<'b, T: DeviceValue> ShaderArgs<'b, Vulkan> for Arc<VulkanBuffer<T>> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut VulkanDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch
            .bindings
            .push((binding, self.buffer(), self.size()));
        Ok(())
    }
}

impl<'b> ShaderArgs<'b, Vulkan> for VulkanBufferSlice {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        _name: &str,
        dispatch: &mut VulkanDispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        dispatch.bindings.push((binding, self.buffer, self.size));
        Ok(())
    }
}

impl<T: DeviceValue> crate::backend::Buffer<Vulkan, T> for Arc<VulkanBuffer<T>> {
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

    fn slice(&self, range: impl RangeBounds<usize>) -> VulkanBufferSlice {
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

        VulkanBufferSlice {
            buffer: self.buffer(),
            _offset: start,
            size: end - start,
        }
    }

    fn usage(&self) -> BufferUsages {
        self.usage
    }
}

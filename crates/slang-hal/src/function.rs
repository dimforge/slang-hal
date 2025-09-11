use crate::backend::{Backend, Dispatch, DispatchGrid, ShaderBinding};
use crate::shader::ShaderArgs;
use minislang::{SlangCompiler, SlangProgram};

struct ShaderArgsDesc {
    buffers: Vec<(String, ShaderBinding)>,
}

// TODO: find a better name… "GpuFunction" perhaps?
pub struct GpuFunction<B: Backend> {
    block_dim: [u32; 3],
    args: ShaderArgsDesc,
    function: B::Function,
}

impl<B: Backend> GpuFunction<B> {
    pub const MAX_NUM_WORKGROUPS: u32 = 65535;

    pub fn from_file(
        backend: &B,
        compiler: &SlangCompiler,
        path: &str,
        entry_point_name: &str,
    ) -> Result<Self, B::Error> {
        let program = compiler.compile(path, B::TARGET, Some(entry_point_name), &[]);
        let module_bytes = program.target_code(0).unwrap();
        let module = backend.load_module_bytes(module_bytes.as_slice())?;
        let function = backend.load_function(&module, entry_point_name)?;
        Self::from_function(entry_point_name, &program, function)
    }

    fn from_function(
        entry_point_name: &str,
        program: &SlangProgram,
        function: B::Function,
    ) -> Result<Self, B::Error> {
        let shader = program.layout(0).unwrap();
        let entry_point = shader.find_entry_point_by_name(entry_point_name).unwrap();
        let block_dim = entry_point.compute_thread_group_size().map(|e| e as u32);
        let mut buffers = vec![];

        for param in entry_point.parameters() {
            let Some(param_var) = param.variable() else {
                continue;
            };
            if param.semantic_name().is_some() {
                continue;
            }
            let binding = ShaderBinding {
                space: param.binding_space(),
                index: param.binding_index(),
            };
            buffers.push((
                param_var
                    .name()
                    // .expect("unnamed parameters not supported yet")
                    .to_string(),
                binding,
            ));
        }

        Ok(Self {
            block_dim,
            args: ShaderArgsDesc { buffers },
            function,
        })
    }

    pub fn block_dim(&self) -> [u32; 3] {
        self.block_dim
    }

    pub fn bind<'a, 'b: 'a>(
        &self,
        dispatch: &mut B::Dispatch<'a>,
        args: &'b impl ShaderArgs<'b, B>,
    ) -> Result<(), B::Error> {
        for (arg_name, arg_binding) in &self.args.buffers {
            args.write_arg(*arg_binding, arg_name, dispatch).unwrap(); // TODO: don't unwrap!
        }
        Ok(())
    }

    /// Launches the function, clamping the dispatch size so it doesn’t exceed WebGPU’s 65535
    /// workgroup count limit.
    ///
    /// Only use this is your shader is capable of handling the case where it should have exceeded
    /// 65535 * WORKGROUP_SIZE.
    ///
    /// Panics if the shader’s block dimension isn’t `1` along the second and third axes, i.e.,
    /// it should be `[anything, 1, 1]`.
    pub fn launch_capped<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        args: &'b impl ShaderArgs<'b, B>,
        num_threads: u32,
    ) -> Result<(), B::Error> {
        assert_eq!(
            self.block_dim[1], 1,
            "launch_capped isn’t applicable in this case"
        );
        assert_eq!(
            self.block_dim[2], 1,
            "launch_capped isn’t applicable in this case"
        );

        // TODO: this is the WebGpu limit. Ideally, we should adjust it depending on the target
        //       platform. And if we do adjust it, the shaders should be adjusted too (because
        //       Slang doesn’t have any way to know the total number of dispatched workgroups).
        let max_num_threads = Self::MAX_NUM_WORKGROUPS * self.block_dim[0];
        self.launch(
            backend,
            pass,
            args,
            [num_threads.min(max_num_threads), 1, 1],
        )
    }

    pub fn launch<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        args: &'b impl ShaderArgs<'b, B>,
        num_threads: [u32; 3],
    ) -> Result<(), B::Error> {
        let grid = [0, 1, 2].map(|i| num_threads[i].div_ceil(self.block_dim[i]));
        self.launch_grid(backend, pass, args, DispatchGrid::Direct(grid))
    }

    pub fn launch_indirect<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        args: &'b impl ShaderArgs<'b, B>,
        grid: &'b B::Buffer<[u32; 3]>,
    ) -> Result<(), B::Error> {
        self.launch_grid(backend, pass, args, DispatchGrid::Indirect(grid))
    }

    pub fn launch_grid<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        args: &'b impl ShaderArgs<'b, B>,
        grid: impl Into<DispatchGrid<'b, B>>,
    ) -> Result<(), B::Error> {
        let mut dispatch = backend.begin_dispatch(pass, &self.function);
        self.bind(&mut dispatch, args)?;
        dispatch.launch(grid, self.block_dim)?;
        Ok(())
    }
}

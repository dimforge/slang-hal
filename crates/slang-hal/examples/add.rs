use minislang::SlangCompiler;
use slang_hal::backend::{Backend, Encoder, WebGpu};
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs, backend::Buffer};
use wgpu::BufferUsages;

// Embed the shaders into the executable for simplicity.
const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/examples/shaders");

#[derive(Shader)]
#[shader(module = "add")]
pub struct GpuAdd<B: Backend> {
    add_assign: GpuFunction<B>,
}

#[derive(ShaderArgs)]
pub struct AddArgs<'a, B: Backend> {
    a: &'a B::Buffer<f32>,
    b: &'a B::Buffer<f32>,
}

impl<B: Backend> GpuAdd<B> {
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        a: &B::Buffer<f32>,
        b: &B::Buffer<f32>,
    ) -> Result<(), B::Error> {
        assert_eq!(a.len(), b.len());

        let args = AddArgs { a, b };
        self.add_assign
            .launch(backend, pass, &args, [a.len() as u32, 1, 1])?;

        Ok(())
    }
}

#[async_std::main]
async fn main() {
    // Initialize the backend and slang compiler.
    #[cfg(feature = "cuda")]
    let backend = Cuda::new().unwrap();
    #[cfg(not(feature = "cuda"))]
    let backend = WebGpu::default().await.unwrap();
    let mut compiler = SlangCompiler::new(vec![]);
    compiler.add_dir(SLANG_SRC_DIR);

    // Run the operation and display the result.
    let a = (0..10000).map(|i| i as f32).collect::<Vec<_>>();
    let b = (0..10000).map(|i| i as f32 * 10.0).collect::<Vec<_>>();
    let result = compute_sum_on_gpu(&backend, &compiler, &a, &b)
        .await
        .unwrap();
    println!("Computed sum: {result:?}");
}

async fn compute_sum_on_gpu<B: Backend>(
    backend: &B,
    compiler: &SlangCompiler,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, B::Error> {
    // Generate the GPU buffers.
    let a = backend.init_buffer(&a, BufferUsages::STORAGE | BufferUsages::COPY_SRC)?;
    let b = backend.init_buffer(&b, BufferUsages::STORAGE)?;

    // Dispatch the operation on the gpu.
    let add = GpuAdd::from_backend(backend, compiler)?;
    let mut encoder = backend.begin_encoding();
    let mut pass = encoder.begin_pass();
    add.launch(backend, &mut pass, &a, &b)?;
    drop(pass);
    backend.submit(encoder)?;

    // Read the result (slow but convenient version).
    backend.slow_read_vec(&a).await
}

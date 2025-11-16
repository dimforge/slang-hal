# slang-hal-build

Build script utilities for compile-time Slang shader compilation.

This crate provides utilities to compile Slang shaders at build time when using the `comptime` feature of `slang-hal`. This eliminates the need for runtime Slang compiler dependency, making it possible to:

- Deploy applications without bundling the Slang compiler
- Support WebAssembly targets
- Reduce application startup time
- Reduce binary dependencies

## Usage

### 1. Add dependencies to your `Cargo.toml`:

```toml
[dependencies]
slang-hal = { version = "0.2", features = ["derive", "comptime"] }

[build-dependencies]
slang-hal-build = "0.2"
```

### 2. Create a `build.rs` file:

```rust
use slang_hal_build::ShaderCompiler;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create shader compiler with output directory
    let out_dir = env::var("OUT_DIR")?;
    let mut compiler = ShaderCompiler::new(vec![], out_dir);

    // Optional: Add shader include directories
    const SHADER_DIR: include_dir::Dir<'_> =
        include_dir::include_dir!("$CARGO_MANIFEST_DIR/shaders");
    compiler.add_dir(SHADER_DIR);

    // Compile your shaders
    // The shader path should be relative to CARGO_MANIFEST_DIR
    compiler.compile_shader("shaders/add.slang", "add_assign", &[])?;

    // With link-time specialization:
    // compiler.compile_shader("shaders/matmul.slang", "matmul", &["specialization::module".to_string()])?;

    Ok(())
}
```

### 3. Use the shaders in your code:

```rust
use slang_hal::{Shader, backend::WebGpu};
use slang_hal::function::GpuFunction;
use minislang::SlangCompiler;

#[derive(Shader)]
#[shader(module = "add")]
pub struct GpuAdd<B: Backend> {
    add_assign: GpuFunction<B>,
}

#[async_std::main]
async fn main() {
    let backend = WebGpu::default().await.unwrap();

    // When comptime is enabled, the compiler parameter is not used
    // (but still required for API compatibility)
    let compiler = SlangCompiler::new(vec![]);

    // The shader is loaded from precompiled data embedded in the binary
    let add = GpuAdd::from_backend(&backend, &compiler).unwrap();

    // Use the shader as normal...
}
```

## How It Works

1. **Build Time**: The build script compiles shaders for all enabled backends (WebGPU, Metal, Vulkan, CUDA, CPU) based on your Cargo features.

2. **Generated Files**: For each shader entry point, the build script generates:
   - Compiled shader source or bytecode (`.wgsl`, `.metal`, `.spv`, `.ptx`, `.dll`)
   - Reflection metadata (Rust source files with shader parameter info)

3. **Compile Time**: The `#[derive(Shader)]` macro generates code that uses `include_bytes!()` and `include!()` to embed the precompiled shader data directly into your binary.

4. **Runtime**: Shaders are loaded instantly from embedded data, with no need to invoke the Slang compiler.

## Backend-Specific Compilation

The build script automatically compiles shaders only for backends enabled via Cargo features:

```toml
[dependencies]
slang-hal = { version = "0.2", features = ["derive", "comptime", "metal", "vulkan"] }
```

This will compile shaders for WebGPU (default), Metal, and Vulkan, but not CUDA or CPU.

## Limitations

- **No runtime specialization**: When `comptime` is enabled, you cannot use runtime shader specialization via `Shader::with_specializations()`. Link-time specialization specified in the `#[shader(specialize = [...])]` attribute is still supported.

- **Larger binaries**: Since shader bytecode for all enabled backends is embedded in the binary, this increases binary size compared to runtime compilation.

- **Longer compile times**: Shaders are compiled during your project's build process, which increases build time.

## WebAssembly Support

The primary motivation for the `comptime` feature is to enable WebAssembly support. The Slang compiler has native dependencies that are difficult to compile for WASM targets. With `comptime`, your WASM binary contains only the precompiled WGSL shaders, eliminating the need for the Slang compiler at runtime.

```toml
[dependencies]
slang-hal = { version = "0.2", features = ["derive", "comptime"] }
```

Then build for WASM:

```bash
cargo build --target wasm32-unknown-unknown
```

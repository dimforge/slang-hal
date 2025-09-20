# slang-hal - A hardware abstraction for Slang

The **slang-hal** library provides abstractions for running slang shaders on any platform supported by the slang compiler.

> **Warning**
**slang-hal** is still very incomplete and under heavy development and is lacking a lot of features and backends.

### Using Slang

In order to compile and run any slang project, be sure to define the `SLANG_DIR` environment variable:
1. Download the Slang compiler libraries for your platform: https://github.com/shader-slang/slang/releases/tag/v2025.16
2. Unzip the downloaded directory, and use its path as value to the `SLANG_DIR` environment variable: `SLANG_DIR=/path/to/slang`.
   Note that the variable must point to the root of the slang installation (i.e. the directory that contains `bin` and `lib`).
   We recommend adding that as a system-wide environment variables so that it also becomes available to your IDE.

### Supported backends

**slang-hal** exposes a unified API for interacting with the GPU in a backend-agnostic way.

| Backend | Shader compilation | Compute pipelines | Render pipelines | Buffer read/write   | Non-Pod types | Indirect dispatch | GPU timestamps | Link-time specialization | 
|---------|--------------------|-------------------|------------------|---------------------|---------------|-------------------|----------------|-------------------------|
| WebGpu  | ✅                 | ✅                 | ❌                 | ✅                   | ✅             | ✅                |  ❌              | ❌ |
| Cuda    | ✅                 | ✅                 | ❌                 | ✅                   | ❌             | ❌                |  ❌              | ❌ |
| Vulkan  | ❌                 | ❌                 | ❌                 | ❌                   | ❌             | ❌                |  ❌              | ❌ |
| Metal   | ❌                 | ❌                 | ❌                 | ❌                   | ❌             | ❌                |  ❌              | ❌ |
| DirectX | ❌                 | ❌                 | ❌                 | ❌                   | ❌              | ❌                | ❌               | ❌ |
| CPU     | ❌                 | ❌                 | ❌                 | ❌                   | ❌              | ❌                | ❌               | ❌ |
| PyTorch | ❌                 | ❌                 | ❌                 | ❌                   | ❌              | ❌                | ❌               | ❌ |
| OptiX   | ❌                 | ❌                 | ❌                 | ❌                   | ❌              | ❌                | ❌               | ❌ |
| OpenCL  | ❌                 | ❌                 | ❌                 | ❌                   | ❌              | ❌                | ❌               | ❌ |

### Other features

**slang-hal** also provides utilities for:
- Writing device-side gpu code in a backend-agnostic way, with the ability to reuse the same code on multiple backend
  even within the same executable.
- Sharing Slang shaders across Rust crates (directly through `cargo`. No need to deal with include paths).
- Checking slang shader validity at compile-time (i.e. when running `cargo build`, `cargo check`, etc.)
- Generating boilerplate and helper functions for loading a shader from Rust and launching its compute pipeline.

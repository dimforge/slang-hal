#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]
#![allow(clippy::result_large_err)]

// Warn users if they enable both comptime and runtime
#[cfg(all(feature = "comptime", feature = "runtime"))]
compile_error!(
    "The 'comptime' and 'runtime' features are mutually exclusive. Use '--no-default-features --features comptime' to enable compile-time shader compilation without runtime dependencies."
);

#[cfg(not(any(feature = "comptime", feature = "runtime")))]
compile_error!("Exactly one of the 'comptime' or 'runtime' features must be enabled.");

pub mod backend;

pub mod function;
pub mod shader;
// mod kernel;

pub use shader::{Shader, ShaderArgs, SlangCompiler};
#[cfg(feature = "derive")]
pub use slang_hal_derive::*;

/// Third-party modules re-exports.
pub mod re_exports {
    pub use bytemuck;
    pub use encase;
    pub use include_dir;
    #[cfg(feature = "runtime")]
    pub use minislang;
    pub use paste;
    #[cfg(feature = "webgpu")]
    pub use wgpu::{self, Device};
}

// Re-export our own BufferUsages type
pub use backend::BufferUsages;
#[cfg(feature = "webgpu")]
pub use backend::{GpuTimestamps, GpuTimingResult};

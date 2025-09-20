#![doc = include_str!("../README.md")]
// #![warn(missing_docs)]
#![allow(clippy::result_large_err)]

pub mod backend;

pub mod function;
pub mod shader;
// mod kernel;

pub use shader::{Shader, ShaderArgs};
#[cfg(feature = "derive")]
pub use slang_hal_derive::*;

/// Third-party modules re-exports.
pub mod re_exports {
    pub use bytemuck;
    pub use encase;
    pub use include_dir;
    pub use minislang;
    pub use paste;
    pub use wgpu::{self, Device};
}

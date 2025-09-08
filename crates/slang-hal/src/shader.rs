use crate::backend::{Backend, ShaderBinding};
use minislang::SlangCompiler;

pub trait Shader<B: Backend>: Sized + 'static {
    /// Instantiates `Self` and all its compute functions from a backend.
    fn from_backend(b: &B, compiler: &SlangCompiler) -> Result<Self, B::Error>;
}

#[derive(thiserror::Error, Debug)]
pub enum ShaderArgsError {
    #[error("argument not found: {0}")]
    ArgNotFound(String),
}

pub trait ShaderArgs<'b, B: Backend> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        name: &str,
        dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a;
}

impl<'b, B: Backend> ShaderArgs<'b, B> for () {
    fn write_arg<'a>(
        &'b self,
        _binding: ShaderBinding,
        name: &str,
        _dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        Err(ShaderArgsError::ArgNotFound(name.to_owned()))
    }
}

impl<'b, B: Backend, T: ShaderArgs<'b, B>> ShaderArgs<'b, B> for Option<T> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        name: &str,
        dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match self {
            Some(arg) => arg.write_arg(binding, name, dispatch),
            None => Err(ShaderArgsError::ArgNotFound(name.to_owned())),
        }
    }
}

impl<'b, B: Backend, T: ShaderArgs<'b, B>> ShaderArgs<'b, B> for &'b T {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        name: &str,
        dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        (*self).write_arg(binding, name, dispatch)
    }
}

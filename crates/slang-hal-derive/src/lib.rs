//! Derive proc-macros for `slang-hal`.

extern crate proc_macro;

use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DataStruct};

#[derive(FromDeriveInput, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersParams {
    pub module: String,
}

#[proc_macro_derive(Shader, attributes(shader))]
pub fn derive_shader(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    let derive_shaders = match DeriveShadersParams::from_derive_input(&input) {
        Ok(v) => v,
        Err(e) => {
            return e.write_errors().into();
        }
    };

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            /*
             * Field attributes.
             */
            let mut kernels_to_build = vec![];
            let slang_path = derive_shaders.module.replace("::", "/");

            for field in fields.iter() {
                let ident = field
                    .ident
                    .as_ref()
                    .expect("unnamed fields not supported")
                    .into_token_stream();

                kernels_to_build.push(quote! {
                    #ident: GpuFunction::from_file(backend, compiler, #slang_path, stringify!(#ident))?,
                });
            }

            let from_backend = quote! {
                Ok(Self {
                    #(
                        #kernels_to_build
                    )*
                })
            };

            quote! {
                #[automatically_derived]
                impl<B: Backend> slang_hal::shader::Shader<B> for #struct_identifier<B> {
                    fn from_backend(backend: &B, compiler: &slang_hal::re_exports::minislang::SlangCompiler) -> Result<Self, B::Error> {
                        #from_backend
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
    .into()
}

#[proc_macro_derive(ShaderArgs)]
pub fn derive_shader_args(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    match &input.data {
        Data::Struct(DataStruct { fields, .. }) => {
            /*
             * Field attributes.
             */
            let mut fields_to_match = vec![];

            for field in fields.iter() {
                let ident = field
                    .ident
                    .as_ref()
                    .expect("unnamed fields not supported")
                    .into_token_stream();

                fields_to_match.push(quote! {
                    stringify!(#ident) => self.#ident.write_arg(binding, name, dispatch)?,
                });
            }

            quote! {
                #[automatically_derived]
                // TODO: don't hard-code the lifetime requirement?
                impl<'b, B: Backend> slang_hal::shader::ShaderArgs<'b, B> for #struct_identifier<'_, B> {
                    fn write_arg<'a>(&'b self, binding: slang_hal::backend::ShaderBinding, name: &str, dispatch: &mut B::Dispatch<'a>) -> Result<(), slang_hal::shader::ShaderArgsError>
                    where 'b: 'a {
                        use slang_hal::backend::Dispatch;
                        match name {
                            #(
                                #fields_to_match
                            )*
                            _ => return Err(slang_hal::shader::ShaderArgsError::ArgNotFound(name.to_owned())),
                        }

                        Ok(())
                    }
                }
            }
        }
        _ => unimplemented!(),
    }
        .into()
}

//! Derive proc-macros for `slang-hal`.

extern crate proc_macro;

use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Data, DataStruct, LitStr};

#[derive(FromDeriveInput, Clone)]
#[darling(attributes(shader))]
struct DeriveShadersParams {
    pub module: String,
    #[darling(default)]
    // Allow dead-code to keep the same derive signature for both comptime and runtime.
    #[cfg_attr(feature = "comptime", allow(dead_code))]
    pub specialize: Option<Vec<LitStr>>,
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
            #[cfg(feature = "comptime")]
            {
                generate_comptime_impl(&input, struct_identifier, &derive_shaders, fields)
            }
            #[cfg(not(feature = "comptime"))]
            {
                generate_runtime_impl(&input, struct_identifier, &derive_shaders, fields)
            }
        }
        _ => unimplemented!(),
    }
    .into()
}

#[cfg(not(feature = "comptime"))]
fn generate_runtime_impl(
    _input: &syn::DeriveInput,
    struct_identifier: &syn::Ident,
    derive_shaders: &DeriveShadersParams,
    fields: &syn::Fields,
) -> proc_macro2::TokenStream {
    /*
     * Field attributes.
     */
    let mut kernels_to_build = vec![];
    let slang_path = derive_shaders.module.replace("::", "/");
    let specialization_paths: Vec<_> = derive_shaders
        .specialize
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|str| str.value().replace("::", "/"))
        .collect();

    for field in fields.iter() {
        let ident = field
            .ident
            .as_ref()
            .expect("unnamed fields not supported")
            .into_token_stream();

        // NOTE: if the user provided specializations explicitly, we use that instead of the default one in the #[shader(...)] definition.
        //       This implies that the user must overwrite **all** the specializations. We don't support partial overwrites yet.
        kernels_to_build.push(quote! {
            #ident: if specializations.is_empty() {
                GpuFunction::from_file(backend, compiler, #slang_path, stringify!(#ident), &[#((#specialization_paths).to_string()),*])?
            } else {
                GpuFunction::from_file(backend, compiler, #slang_path, stringify!(#ident), specializations)?
            },
        });
    }

    let with_specializations_impl = quote! {
        Ok(Self {
            #(
                #kernels_to_build
            )*
        })
    };

    quote! {
        #[automatically_derived]
        impl<B: Backend> slang_hal::shader::Shader<B> for #struct_identifier<B> {
            fn with_specializations(backend: &B, compiler: &slang_hal::SlangCompiler, specializations: &[String]) -> Result<Self, B::Error> {
                #with_specializations_impl
            }
        }
    }
}

#[cfg(feature = "comptime")]
fn generate_comptime_impl(
    _input: &syn::DeriveInput,
    struct_identifier: &syn::Ident,
    _derive_shaders: &DeriveShadersParams,
    fields: &syn::Fields,
) -> proc_macro2::TokenStream {
    let mut field_initializers = vec![];

    for field in fields.iter() {
        let field_ident = field.ident.as_ref().expect("unnamed fields not supported");
        let entry_point = field_ident.to_string();

        // Generate code to load precompiled data from build script output
        // Convert module path from "::" to "/" for file system paths
        let module_path = _derive_shaders.module.replace("::", "/");

        field_initializers.push(quote! {
            #field_ident: {
                use slang_hal::backend::CompileTarget;

                let data = match B::TARGET {
                    #[cfg(feature = "webgpu")]
                    CompileTarget::Wgsl => {
                        include!(concat!(env!("OUT_DIR"), "/", #module_path, "/", #entry_point, "/", #entry_point, "_webgpu_reflection.rs"))
                    }
                    #[cfg(feature = "metal")]
                    CompileTarget::Metal => {
                        include!(concat!(env!("OUT_DIR"), "/", #module_path, "/", #entry_point, "/", #entry_point, "_metal_reflection.rs"))
                    }
                    #[cfg(feature = "vulkan")]
                    CompileTarget::Spirv => {
                        include!(concat!(env!("OUT_DIR"), "/", #module_path, "/", #entry_point, "/", #entry_point, "_vulkan_reflection.rs"))
                    }
                    #[cfg(feature = "cuda")]
                    CompileTarget::Ptx => {
                        include!(concat!(env!("OUT_DIR"), "/", #module_path, "/", #entry_point, "/", #entry_point, "_cuda_reflection.rs"))
                    }
                    #[cfg(feature = "cpu")]
                    CompileTarget::HostHostCallable => {
                        include!(concat!(env!("OUT_DIR"), "/", #module_path, "/", #entry_point, "/", #entry_point, "_cpu_reflection.rs"))
                    }
                    _ => panic!("No precompiled shader data available for backend target {:?}", B::TARGET),
                };

                #[allow(unreachable_code)] // If none of the backend features are enabled this will be unreachable.
                { GpuFunction::from_precompiled(backend, &data)? }
            }
        });
    }

    quote! {
        #[automatically_derived]
        impl<B: Backend> slang_hal::shader::Shader<B> for #struct_identifier<B> {
            fn with_specializations(backend: &B, _compiler: &slang_hal::SlangCompiler, specializations: &[String]) -> Result<Self, B::Error> {
                if !specializations.is_empty() {
                    panic!("Runtime specializations are not supported with the 'comptime' feature. Use link-time specialization in the #[shader(specialize = [...])] attribute instead.");
                }

                Ok(Self {
                    #(
                        #field_initializers,
                    )*
                })
            }
        }
    }
}

#[proc_macro_derive(ShaderArgs)]
pub fn derive_shader_args(item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::DeriveInput);
    let struct_identifier = &input.ident;

    // Extract generics from the input struct
    let mut generics = input.generics.clone();

    // Add 'b lifetime to impl generics
    generics.params.insert(0, syn::parse_quote!('b));

    // Add B: Backend bound to where clause
    let where_clause = generics.make_where_clause();
    where_clause.predicates.push(syn::parse_quote!(B: Backend));

    let (impl_generics, _, _) = generics.split_for_impl();
    let (_, ty_generics, _) = input.generics.split_for_impl();

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

            let where_clause = &generics.where_clause;

            quote! {
                #[automatically_derived]
                impl #impl_generics slang_hal::shader::ShaderArgs<'b, B> for #struct_identifier #ty_generics
                #where_clause
                {
                    fn write_arg<'c>(&'b self, binding: slang_hal::backend::ShaderBinding, name: &str, dispatch: &mut B::Dispatch<'c>) -> Result<(), slang_hal::shader::ShaderArgsError>
                    where 'b: 'c {
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

//! Build script utilities for compile-time Slang shader compilation.
//!
//! This crate provides utilities to compile Slang shaders at build time,
//! enabling the `comptime` feature of `slang-hal` which eliminates the need
//! for runtime Slang compiler dependency.

use minislang::SlangCompiler;
use minislang::shader_slang::CompileTarget;
use std::fs;
use std::path::{Path, PathBuf};

/// Information about a backend to compile for.
struct BackendInfo {
    name: &'static str,
    target: CompileTarget,
    feature: &'static str,
    extension: &'static str,
}

const BACKENDS: &[BackendInfo] = &[
    BackendInfo {
        name: "webgpu",
        target: CompileTarget::Wgsl,
        feature: "webgpu",
        extension: "wgsl",
    },
    BackendInfo {
        name: "metal",
        target: CompileTarget::Metal,
        feature: "metal",
        extension: "metal",
    },
    BackendInfo {
        name: "vulkan",
        target: CompileTarget::Spirv,
        feature: "vulkan",
        extension: "spv",
    },
    BackendInfo {
        name: "cuda",
        target: CompileTarget::Ptx,
        feature: "cuda",
        extension: "ptx",
    },
    BackendInfo {
        name: "cpu",
        target: CompileTarget::HostHostCallable,
        feature: "cpu",
        extension: "dll",
    },
];

/// Configuration for shader compilation.
pub struct ShaderCompiler {
    compiler: SlangCompiler,
    out_dir: PathBuf,
}

impl ShaderCompiler {
    /// Creates a new shader compiler with the given search paths.
    ///
    /// The `out_dir` should typically be `std::env::var("OUT_DIR")` from your build script.
    pub fn new(search_paths: Vec<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        Self {
            compiler: SlangCompiler::new(search_paths),
            out_dir: out_dir.into(),
        }
    }

    /// Adds a directory of shader files to the compiler's search path.
    ///
    /// This is useful for making shader imports work correctly.
    pub fn add_dir(&mut self, dir: include_dir::Dir<'static>) {
        self.compiler.add_dir(dir);
    }

    /// Sets a global preprocessor macro for all shader compilations.
    pub fn set_global_macro(&mut self, name: impl ToString, value: impl ToString) {
        self.compiler.set_global_macro(name, value);
    }

    /// Compiles all entry points from all .slang shader files in a directory (recursively).
    ///
    /// # Arguments
    ///
    /// * `shader_dir` - Path to the directory containing .slang shader files (relative to CARGO_MANIFEST_DIR)
    /// * `specializations` - Optional link-time specialization modules
    ///
    /// # Returns
    ///
    /// Returns a list of (shader_path, entry_point_name) pairs that were compiled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In your build.rs:
    /// let mut compiler = ShaderCompiler::new(vec![], env::var("OUT_DIR")?);
    /// compiler.compile_shaders_dir("examples/shaders", &[])?;
    /// ```
    pub fn compile_shaders_dir(
        &self,
        shader_dir: impl AsRef<Path>,
        specializations: &[String],
    ) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
        let shader_dir = shader_dir.as_ref();
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let full_dir = Path::new(&manifest_dir).join(shader_dir);

        let mut compiled = Vec::new();

        // Recursively find all .slang files
        for entry in walkdir::WalkDir::new(&full_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("slang"))
        {
            let shader_path = entry.path();
            let relative_path = shader_path
                .strip_prefix(&manifest_dir)
                .unwrap_or(shader_path)
                .to_str()
                .ok_or("Invalid shader path")?;

            // Compute module path from shader file path
            // Remove the shader_dir prefix and .slang extension
            let module_path = shader_path
                .strip_prefix(&full_dir)
                .unwrap_or(shader_path)
                .with_extension("");
            let module_path = module_path.to_str().ok_or("Invalid module path")?;

            // Find all entry points in this shader
            let entry_points = self.find_entry_points(relative_path)?;

            // Compile each entry point
            for entry_point in entry_points {
                self.compile_shader_entry_point(
                    relative_path,
                    module_path,
                    &entry_point,
                    specializations,
                )?;
                compiled.push((relative_path.to_string(), entry_point));
            }
        }

        Ok(compiled)
    }

    /// Compiles a single shader entry point for all enabled backends.
    ///
    /// # Arguments
    ///
    /// * `shader_path` - Path to the .slang shader file (relative to CARGO_MANIFEST_DIR)
    /// * `entry_point` - Name of the entry point function to compile
    /// * `specializations` - Optional link-time specialization modules
    ///
    /// # Returns
    ///
    /// Returns the identifier to use in the generated code (derived from entry_point name).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In your build.rs:
    /// let mut compiler = ShaderCompiler::new(vec![], env::var("OUT_DIR")?);
    /// compiler.compile_shader("examples/shaders/add.slang", "add_assign", &[])?;
    /// ```
    pub fn compile_shader(
        &self,
        shader_path: impl AsRef<Path>,
        entry_point: &str,
        specializations: &[String],
    ) -> Result<String, Box<dyn std::error::Error>> {
        let shader_path = shader_path.as_ref();
        let shader_path_str = shader_path.to_str().ok_or("Invalid shader path")?;

        // Compute module path from shader file path
        // Remove .slang extension and convert to module path
        let module_path_buf = shader_path.with_extension("");
        let module_path = module_path_buf
            .file_stem()
            .ok_or("Invalid shader path")?
            .to_str()
            .ok_or("Invalid module path")?;

        self.compile_shader_entry_point(
            shader_path_str,
            module_path,
            entry_point,
            specializations,
        )?;
        Ok(entry_point.to_string())
    }

    /// Finds all entry points in a shader file.
    fn find_entry_points(
        &self,
        shader_path: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // Compile the shader without specifying an entry point to get all entry points
        let program = self.compiler.compile(
            shader_path,
            CompileTarget::Wgsl, // Use any target just for reflection
            None,                // No specific entry point - get all
            &[],
            &[],
        );

        let layout = program
            .layout(0)
            .map_err(|e| format!("Failed to get shader layout: {:?}", e))?;

        let mut entry_points = Vec::new();
        for i in 0..layout.entry_point_count() {
            if let Some(ep) = layout.entry_point_by_index(i) {
                entry_points.push(ep.name().to_string());
            }
        }

        Ok(entry_points)
    }

    /// Internal method to compile a specific shader entry point.
    fn compile_shader_entry_point(
        &self,
        shader_path: &str,
        module_path: &str,
        entry_point: &str,
        specializations: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create output directory for this entry point using module path
        // This avoids conflicts when multiple modules have the same entry point name
        let entry_out_dir = self.out_dir.join(module_path).join(entry_point);
        fs::create_dir_all(&entry_out_dir)?;

        // Convert specializations to paths
        let specialization_paths: Vec<String> = specializations
            .iter()
            .map(|s| s.replace("::", "/"))
            .collect();

        // Compile for each enabled backend
        for backend_info in BACKENDS {
            // Check if this backend should be compiled
            if !should_compile_for_backend(backend_info.feature) {
                continue;
            }

            println!("cargo:rerun-if-changed={}", shader_path);

            // Compile the shader
            let program = self.compiler.compile(
                shader_path,
                backend_info.target,
                Some(entry_point),
                &specialization_paths,
                &[],
            );

            // Extract compiled bytes
            let blob = program.target_code(0).map_err(|e| {
                format!(
                    "Failed to get target code for {}: {:?}",
                    backend_info.name, e
                )
            })?;
            let module_bytes = blob.as_slice();

            // Write to output file
            let output_filename = format!(
                "{}_{}.{}",
                entry_point, backend_info.name, backend_info.extension
            );
            let output_path = entry_out_dir.join(&output_filename);
            fs::write(&output_path, module_bytes)?;

            // Extract and write reflection metadata
            let reflection = extract_reflection(&program, entry_point)?;
            let reflection_path = entry_out_dir.join(format!(
                "{}_{}_reflection.rs",
                entry_point, backend_info.name
            ));
            let reflection_code =
                generate_reflection_code(&reflection, module_path, entry_point, &output_filename);
            fs::write(&reflection_path, reflection_code)?;
        }

        Ok(())
    }
}

/// Shader reflection metadata extracted at compile-time.
struct ShaderReflection {
    block_dim: [u32; 3],
    buffers: Vec<(String, ShaderBinding)>,
}

struct ShaderBinding {
    space: u32,
    index: u32,
}

/// Extracts reflection metadata from a compiled shader program.
fn extract_reflection(
    program: &minislang::SlangProgram,
    entry_point_name: &str,
) -> Result<ShaderReflection, String> {
    let shader = program
        .layout(0)
        .map_err(|e| format!("Failed to get shader layout: {:?}", e))?;

    let entry_point = shader
        .find_entry_point_by_name(entry_point_name)
        .ok_or_else(|| format!("Entry point '{}' not found", entry_point_name))?;

    let block_dim = entry_point.compute_thread_group_size().map(|e| e as u32);

    let mut buffers = Vec::new();
    for param in entry_point.parameters() {
        let Some(param_var) = param.variable() else {
            continue;
        };
        // Skip semantic parameters (like SV_DispatchThreadID)
        if param.semantic_name().is_some() {
            continue;
        }

        let binding = ShaderBinding {
            space: param.binding_space(),
            index: param.binding_index(),
        };

        buffers.push((param_var.name().to_string(), binding));
    }

    Ok(ShaderReflection { block_dim, buffers })
}

/// Generates Rust code for reflection metadata.
fn generate_reflection_code(
    reflection: &ShaderReflection,
    module_path: &str,
    entry_point: &str,
    backend_filename: &str,
) -> String {
    let block_dim = reflection.block_dim;
    let buffers: Vec<String> = reflection
        .buffers
        .iter()
        .map(|(name, binding)| {
            format!(
                "(\"{}\".to_string(), slang_hal::backend::ShaderBinding {{ space: {}, index: {} }})",
                name, binding.space, binding.index
            )
        })
        .collect();

    format!(
        r#"// Auto-generated reflection metadata
slang_hal::function::PrecompiledShaderData {{
    module_bytes: include_bytes!(concat!(env!("OUT_DIR"), "/{}/{}/{}")),
    entry_point: "{}",
    block_dim: [{}, {}, {}],
    buffers: vec![{}],
}}
"#,
        module_path,
        entry_point,
        backend_filename,
        entry_point,
        block_dim[0],
        block_dim[1],
        block_dim[2],
        buffers.join(", ")
    )
}

/// Checks if we should compile for a specific backend based on cargo features.
fn should_compile_for_backend(feature: &str) -> bool {
    // Check if the feature is enabled via CARGO_FEATURE_* environment variable
    let feature_var = format!("CARGO_FEATURE_{}", feature.to_uppercase());
    std::env::var(&feature_var).is_ok()
}

use include_dir::Dir;
pub use shader_slang;
use shader_slang::{
    CompileTarget, CompilerOptions, Downcast, GlobalSession, OptimizationLevel, SessionDesc,
    TargetDesc,
};
pub use shader_slang_sys;
use std::ffi::CString;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

mod dir;

// TODO: refactor to a separate crate? Or import slang-hal?
pub struct SlangCompiler {
    session: GlobalSession,
    search_paths: Vec<PathBuf>,
    global_macros: Vec<(String, String)>,
    tmp: TempDir,
}

pub struct SlangProgram {
    #[allow(dead_code)]
    session: shader_slang::Session,
    program: shader_slang::ComponentType,
}

impl Deref for SlangProgram {
    type Target = shader_slang::ComponentType;
    fn deref(&self) -> &Self::Target {
        &self.program
    }
}

impl SlangCompiler {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            session: GlobalSession::new().unwrap(),
            search_paths,
            global_macros: Vec::new(),
            tmp: tempfile::tempdir().unwrap(),
        }
    }

    pub fn add_dir(&mut self, dir: Dir<'static>) {
        dir::write_dir_to_disk(&self.tmp, &dir);
    }

    pub fn set_global_macro(&mut self, name: impl ToString, value: impl ToString) {
        self.global_macros
            .push((name.to_string(), value.to_string()));
    }

    pub fn compile(
        &self,
        module: &str,
        target: CompileTarget,
        entry_point: Option<&str>,
        macro_defines: &[(String, String)],
    ) -> SlangProgram {
        let (linked_program, session) = {
            let search_paths: Vec<_> = self
                .search_paths
                .iter()
                .map(|p| p.as_ref())
                .chain(std::iter::once(self.tmp.path()))
                .map(|path| CString::new(path.as_os_str().as_encoded_bytes()).unwrap())
                .collect();

            // All compiler options are available through this builder.
            let mut session_options = CompilerOptions::default()
                .optimization(OptimizationLevel::Maximal)
                .matrix_layout_row(true)
                .macro_define("SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT", "true");

            for (macro_name, macro_val) in macro_defines.iter().chain(&self.global_macros) {
                session_options = session_options.macro_define(macro_name, macro_val);
            }

            let target_desc = TargetDesc::default().format(target);

            let targets = [target_desc];
            let search_paths_ptr: Vec<_> = search_paths.iter().map(|path| path.as_ptr()).collect();

            let session_desc = SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths_ptr)
                .options(&session_options);

            let session = self
                .session
                .create_session(&session_desc)
                .expect("failed to create session");

            let module = session.load_module(module).unwrap();

            let entry_points: Vec<_> = module
                .entry_points()
                .filter(|e| entry_point.is_none() || e.function_reflection().name() == entry_point)
                .map(|e| e.downcast().clone())
                .collect();
            let program = session
                .create_composite_component_type(&entry_points)
                .unwrap();
            let linked_program = program.link().unwrap();
            (linked_program, session)
        };

        SlangProgram {
            program: linked_program,
            session,
        }
    }

    pub fn compile_to(
        &self,
        target: CompileTarget,
        module: &str,
        target_file: impl AsRef<Path>,
        macro_defines: &[(String, String)],
    ) {
        let program = self.compile(module, target, None, macro_defines);
        let code = program
            .program
            .target_code(0)
            .expect("failed to link target code");
        std::fs::write(target_file, code.as_str().unwrap()).unwrap();
    }

    /// Traverses the `src_dir` directory recursively and compile slang files it contains into the
    /// `target_dir`, replicating the same directory hierarchy.
    pub fn compile_all(
        &self,
        target: CompileTarget,
        src_dir: impl AsRef<Path>,
        target_dir: impl AsRef<Path>,
        macro_defines: &[(String, String)],
    ) {
        use walkdir::WalkDir;

        let src_dir = src_dir.as_ref();
        for entry in WalkDir::new(src_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.metadata().unwrap().is_file())
        {
            let path = entry.path();

            if path.extension().unwrap().to_str().unwrap() == "slang" {
                let path_with_modified_ext = path.with_extension(target_extension(target));
                let target_path = path_with_modified_ext.to_str().unwrap();
                let target_path = target_path.replace(
                    src_dir.to_str().unwrap(),
                    target_dir.as_ref().to_str().unwrap(),
                );
                let target_path = PathBuf::from(target_path);
                let target_parent_dir = target_path.parent().unwrap();

                println!(
                    "Compiling {} into {}.",
                    path.display(),
                    target_path.display()
                );
                std::fs::create_dir_all(target_parent_dir).unwrap();
                self.compile_to(target, path.to_str().unwrap(), target_path, macro_defines);
            }
        }
    }
}

fn target_extension(target: CompileTarget) -> &'static str {
    match target {
        CompileTarget::Wgsl => "wgsl",
        CompileTarget::Ptx => "ptx",
        CompileTarget::CudaSource => "cu",
        CompileTarget::Metal => "metal",
        _ => todo!(),
    }
}

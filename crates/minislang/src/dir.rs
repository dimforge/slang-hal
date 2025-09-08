use include_dir::{Dir, DirEntry};
use tempfile::TempDir;

// NOTE: ideally we should load directly from `modules`, from RAM.
//       But that isn’t working too well for now (see commented-out
//       code further below).
pub fn write_dir_to_disk(target_root_dir: &TempDir, modules: &Dir) {
    for entry in modules.entries() {
        match entry {
            DirEntry::Dir(dir) => {
                write_dir_to_disk(target_root_dir, dir);
            }
            DirEntry::File(file) => {
                let path = target_root_dir.path().join(file.path());
                let bytes = file.contents();
                // TODO: won’t work on WASM or restricted environments without
                //       access to the tmp dir.
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, bytes).unwrap();
            }
        }
    }
}

/*
 * TODO: the functions below aim to load the module directly from the executable in RAM
 *       but this crashes the bindings attempting to unwrap a null diagnostics pointer.
 */
// // TODO: handle IR blobs too.
// pub fn register_modules_from_dir(session: &mut Session, modules: &Dir) {
//     let mut prev_num_failures = 1;
//     let mut num_failures = 0;
//
//     while prev_num_failures != num_failures {
//         prev_num_failures = num_failures;
//         num_failures = register_modules_from_dir_recursive(session, modules);
//     }
// }
//
// pub fn register_modules_from_dir_recursive(session: &mut Session, modules: &Dir) -> usize {
//     let mut num_failures = 0;
//     for entry in modules.entries() {
//         match entry {
//             DirEntry::Dir(dir) => {
//                 num_failures += register_modules_from_dir_recursive(session, dir);
//             }
//             DirEntr
//                 let path = file.path().to_string_lossy();
//                 let module_name = file.path().with_extension("").to_string_lossy().replace("/", "/");
//
//                 println!("loading {}", module_name);
//                 let Some(src) = file.contents_utf8() else {
//                     println!("attempted to read module {module_name} at {path} with non-UTF8 content");
//                     continue;
//                 };
//
//                 if let Err(e) = session.load_module_from_source_string(&module_name, &path, &src) {
//                     println!("failed to load module {module_name}");
//                 } else {
//                     println!("successfully loaded module {module_name}: {path}");
//                 }
//             }
//         }
//     }
//
//     num_failures
// }

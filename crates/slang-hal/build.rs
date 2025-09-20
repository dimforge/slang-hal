use std::env;
use std::path::Path;

// Automatically copy the dynamic libraries from the slang dir to the target dir.
fn main() {
    println!("cargo:rerun-if-env-changed=SLANG_LIB_DIR");

    let lib_dir = if let Ok(dir) = env::var("SLANG_LIB_DIR") {
        dir
    } else if let Ok(dir) = env::var("SLANG_DIR") {
        format!("{dir}/lib")
    } else if let Ok(dir) = env::var("VULKAN_SDK") {
        format!("{dir}/lib")
    } else {
        panic!("The environment variable SLANG_LIB_DIR, SLANG_DIR, or VULKAN_SDK must be set");
    };

    let bin_dir = if let Ok(dir) = env::var("SLANG_BIN_DIR") {
        dir
    } else if let Ok(dir) = env::var("SLANG_DIR") {
        format!("{dir}/bin")
    } else if let Ok(dir) = env::var("VULKAN_SDK") {
        format!("{dir}/bin")
    } else {
        panic!("The environment variable SLANG_BIN_DIR, SLANG_DIR, or VULKAN_SDK must be set");
    };

    if !lib_dir.is_empty() {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }

    let out_dir = env::var("OUT_DIR").expect("Couldn't determine output directory.");
    let out_dir = Path::new(&out_dir);
    let cpy_target = out_dir.join("../../..");

    dircpy::copy_dir(&lib_dir, &cpy_target).unwrap_or_else(|e| panic!("could not copy dynamic libraries from `{lib_dir:?}` to target directory `{cpy_target:?}`: {e}"));
    dircpy::copy_dir(&bin_dir, &cpy_target).unwrap_or_else(|e| panic!("could not copy dynamic libraries from `{bin_dir:?}` to target directory `{cpy_target:?}`: {e}"));
}

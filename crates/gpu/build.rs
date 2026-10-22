fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_OXIDE_TARGET");
    println!("cargo:rustc-check-cfg=cfg(bloomery_nvfp4_mma)");
    if matches!(
        std::env::var("CUDA_OXIDE_TARGET").as_deref(),
        Ok("sm_120a" | "sm_120f" | "sm_121a" | "sm_121f")
    ) {
        println!("cargo:rustc-cfg=bloomery_nvfp4_mma");
    }
}

//! Compute backend abstraction powered by `cubecl`.
//!
//! This module owns a single shared [`Client`] and selects the concrete backend
//! ([`BackendRuntime`]/[`Device`]) via cargo features (`wgpu`, `cuda` or `cpu`).
//!
//! Individual `cube` kernels live in the modules that use them (e.g. `matching`,
//! `mvs`) rather than here, so the same kernel source runs unchanged on the CPU,
//! WGSL and CUDA backends.

use std::sync::{Mutex, OnceLock};

use cubecl::client::ComputeClient;
use cubecl::Runtime;

#[cfg(feature = "wgpu")]
pub use cubecl::wgpu::{WgpuDevice as Device, WgpuRuntime as BackendRuntime};

#[cfg(all(not(feature = "wgpu"), feature = "cuda"))]
pub use cubecl::cuda::{CudaDevice as Device, CudaRuntime as BackendRuntime};

#[cfg(all(not(feature = "wgpu"), not(feature = "cuda"), feature = "cpu"))]
pub use cubecl::cpu::{CpuDevice as Device, CpuRuntime as BackendRuntime};

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "cpu")))]
compile_error!("至少启用一个后端 feature：`wgpu`、`cuda` 或 `cpu`");

/// Shared compute client for the selected backend.
pub type Client = ComputeClient<BackendRuntime>;

static CLIENT: OnceLock<Mutex<Option<Client>>> = OnceLock::new();

fn client_slot() -> &'static Mutex<Option<Client>> {
    CLIENT.get_or_init(|| Mutex::new(None))
}

/// 惰性初始化并返回全局 compute client（克隆句柄，底层共享同一设备）。
///
/// 返回可克隆的 [`Client`]（内部为 Arc），因此即使之后调用 [`reset_client`]
/// 丢弃底层设备，已持有的句柄也会在各自 drop 时正常释放。
pub fn client() -> Client {
    let mut slot = client_slot().lock().unwrap();
    if slot.is_none() {
        let device = Device::default();
        *slot = Some(BackendRuntime::client(&device));
    }
    slot.as_ref().unwrap().clone()
}

/// 丢弃当前设备，强制下一次 [`client`] 调用重建。
///
/// 用于设备丢失（TDR / 驱动重置）后的恢复：已丢失的设备无法再提交，
/// 唯一办法是丢弃重建。注意：仅当没有其它线程正在并发使用该设备时才安全
/// （全分辨率下 MVS 并行度=1，天然满足）。
pub fn reset_client() {
    let mut slot = client_slot().lock().unwrap();
    *slot = None;
}

/// Initialize the backend (and fail early if no device is available).
pub fn init() {
    let c = client();
    println!("Compute backend: {}", BackendRuntime::name(&c));
    #[cfg(not(feature = "cpu"))]
    println!("GPU memory: {} MB", gpu_total_memory_bytes() / 1_000_000);
}

/// 请求后端释放可释放的显存（归还内存池中已 drop 的缓冲给 GPU）。
pub fn memory_cleanup() {
    client().memory_cleanup();
}

/// 返回后端当前的内存占用概要（诊断用，默认开启）。
pub fn memory_usage() -> String {
    match client().memory_usage() {
        Ok(u) => format!(
            "{} MB reserved ({} allocs)",
            u.bytes_reserved / 1_000_000,
            u.number_allocs
        ),
        Err(_) => "memory_usage unavailable".to_string(),
    }
}

/// 探测设备总内存（字节）。CPU 后端即系统内存；GPU 后端为真实显存。
///
/// 探测顺序：环境变量 `OWLRSCON_GPU_MEM_MB`（MB）覆盖 → 平台/后端 API 探测。
/// 探测失败时：CPU 后端回退系统内存（不致命）；GPU 后端报错退出。
fn detect_total_memory() -> Option<usize> {
    #[cfg(feature = "cpu")]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        return Some(sys.total_memory() as usize);
    }
    #[cfg(feature = "cuda")]
    {
        return cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(_free, total)| total);
    }
    #[cfg(feature = "wgpu")]
    {
        return detect_wgpu_memory();
    }
    #[allow(unreachable_code)]
    None
}

/// 缓存探测到的总显存，避免重复探测。
static TOTAL_MEMORY: OnceLock<usize> = OnceLock::new();

/// 返回设备总内存（字节）。
#[allow(dead_code)]
pub fn gpu_total_memory_bytes() -> usize {
    *TOTAL_MEMORY.get_or_init(|| {
        // 环境变量覆盖（MB）
        if let Ok(v) = std::env::var("OWLRSCON_GPU_MEM_MB") {
            if let Ok(mb) = v.trim().parse::<usize>() {
                return mb * 1_000_000;
            }
        }
        match detect_total_memory() {
            Some(total) => total,
            None => {
                #[cfg(feature = "cpu")]
                {
                    let mut sys = sysinfo::System::new();
                    sys.refresh_memory();
                    sys.total_memory() as usize
                }
                #[cfg(not(feature = "cpu"))]
                {
                    eprintln!(
                        "[backend] 无法探测 GPU 显存，请设置环境变量 OWLRSCON_GPU_MEM_MB（单位 MB）后重试"
                    );
                    std::process::exit(1);
                }
            }
        }
    })
}

/// wgpu 后端显存探测（按平台）。
#[cfg(feature = "wgpu")]
fn detect_wgpu_memory() -> Option<usize> {
    #[cfg(target_os = "windows")]
    {
        return dxgi_total_memory();
    }
    #[cfg(target_os = "linux")]
    {
        return sysfs_vram_total();
    }
    // macOS：wgpu 走 Metal。Apple Silicon 为统一内存架构，GPU 与 CPU 共享同一
    // 物理内存，可用显存即系统内存；直接用系统总内存作为预算（Intel 独显机型为
    // 保守上界，且并行度仍受 CPU 核数封顶）。
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        return Some(sys.total_memory() as usize);
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Windows：DXGI 枚举显卡，取最大独立显存（对应 wgpu 通常选中的独显）。
#[cfg(all(feature = "wgpu", target_os = "windows"))]
fn dxgi_total_memory() -> Option<usize> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1};
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut max_mem = 0usize;
        for i in 0..64u32 {
            let adapter: IDXGIAdapter1 = match factory.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => break,
            };
            if let Ok(desc) = adapter.GetDesc1() {
                max_mem = max_mem.max(desc.DedicatedVideoMemory);
            }
        }
        if max_mem > 0 {
            Some(max_mem)
        } else {
            None
        }
    }
}

/// Linux：读 DRM 节点显存总量（字节），取最大值。
#[cfg(all(feature = "wgpu", target_os = "linux"))]
fn sysfs_vram_total() -> Option<usize> {
    let mut max_mem = 0usize;
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let path = entry.path().join("device/mem_info_vram_total");
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(v) = s.trim().parse::<usize>() {
                    max_mem = max_mem.max(v);
                }
            }
        }
    }
    if max_mem > 0 {
        Some(max_mem)
    } else {
        None
    }
}

/// 根据单任务内存占用自动计算可并行的任务数。
///
/// CPU 后端设备即系统内存，可用预算取总内存的 1/2；GPU 后端（wgpu/cuda）取
/// 真实显存的 80%（留 20% 给内存池/驱动开销）。再以 CPU 逻辑核数封顶（至少 1）。
pub fn parallelism(per_task_bytes: usize) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    #[cfg(feature = "cpu")]
    let budget = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        (sys.total_memory() as usize) / 2
    };
    #[cfg(not(feature = "cpu"))]
    let budget = gpu_total_memory_bytes() * 4 / 5;
    (budget / per_task_bytes.max(1)).clamp(1, cpus)
}

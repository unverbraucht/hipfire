# Known Issues → Fix Mapping

## /dev/kfd missing
```bash
# Ubuntu/Debian
sudo apt install linux-firmware amdgpu-dkms
sudo reboot

# Arch Linux  
sudo pacman -S linux-firmware
sudo reboot

# Fedora
sudo dnf install kernel-modules-extra
sudo reboot
```

## ROCm runtime not installed
```bash
# Ubuntu 22.04/24.04
sudo apt install rocm-hip-runtime
# OR just the library:
sudo apt install libamdhip64-dev

# Arch Linux
yay -S rocm-hip-runtime
```

## Pre-compiled kernels missing
```bash
# If hipcc is available (ROCm SDK installed):
./scripts/compile-kernels.sh gfx1010  # for RX 5700 XT
./scripts/compile-kernels.sh gfx1030  # for RX 6800 XT
./scripts/compile-kernels.sh gfx1100  # for RX 7900 XTX
./scripts/compile-kernels.sh gfx1200  # for RX 9070

# If hipcc is NOT available:
# Download from GitHub releases for your arch
```

## Test binaries not built
```bash
cargo build --release --features deltanet \
  --example test_kernels \
  --example test_inference \
  --example infer \
  --example infer_hfq \
  -p engine
```

## OOM during inference
Try a smaller model:
- 4B fits in 4GB VRAM
- 9B needs 6GB+ VRAM
- 8B Qwen3 needs 5GB+ VRAM

Or reduce context: the `--maxgen` flag limits generation length.

## VRAM leak (65MB per model load)
Update to latest version. Fixed in commit 0d1fca6:
- Call `kv_cache.free_gpu(&mut gpu)` before dropping
- Call `gpu.drain_pool()` after unloading

## Slow first inference (cold kernel compilation)
Pre-compiled kernels eliminate this. Verify:
- `kernels/compiled/{your_arch}/` has .hsaco files
- The binary prints "pre-compiled kernels: kernels/compiled/{arch}" at startup

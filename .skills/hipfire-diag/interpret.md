# hipfire-diag: Interpretation Guide

You are interpreting the output of hipfire's GPU diagnostic tool. The output is JSON. Walk the user through what works and what doesn't, then offer to fix any issues.

## Reading the results

### gpu.kfd = false
**No AMD GPU detected.** The user needs:
- An AMD GPU (RX 5700 XT, 6800 XT, 7900 XTX, 9070, or similar)
- The amdgpu kernel driver installed (`sudo apt install linux-firmware` on Ubuntu)
- Check: `ls /dev/kfd` should exist

### gpu.arch = "unknown"
The test binary couldn't detect the GPU architecture. Either:
- The binary isn't built yet → offer to run `cargo build --release --features deltanet --example test_kernels -p engine`
- ROCm runtime isn't installed → `sudo apt install rocm-hip-runtime` (Ubuntu) or equivalent

### kernels.{arch} = 0
No pre-compiled kernels for this GPU architecture. Two options:
1. If the user has hipcc: `./scripts/compile-kernels.sh {arch}`
2. If not: download pre-compiled kernels from the GitHub release for their arch

### kernel_tests.failed > 0
A GPU kernel failed. Look at the `failures` array for specifics:
- **"FAIL: hipcc compilation"** → kernel source error or missing include. Check `kernels/src/turbo_common.h` exists.
- **"FAIL: NaN"** → numerical issue, likely a kernel bug. Report to maintainers with the exact test name.
- **"PANIC"** → GPU hang. Could be VGPR overflow or infinite loop. Report with GPU arch.

### inference_tests.failed > 0
The model loads but inference fails. Common causes:
- **OOM** → model too large for this GPU. Try a smaller model.
- **"VRAM leak"** → known issue if `free_gpu()` isn't called. Update to latest version.
- **Low tok/s** → check if pre-compiled kernels loaded (avoids hipcc cold start penalty).

### inference_tests.tok_s < 10
Something is severely wrong. Expected minimums:
- 4B model: >50 tok/s
- 9B model: >40 tok/s  
- 8B Qwen3: >55 tok/s
If much lower, check for: VRAM contention (other processes using GPU), thermal throttling, or wrong kernel arch.

## Offering fixes

After interpreting, tell the user:
1. What works ✓
2. What doesn't ✗
3. For each issue: "Would you like me to [specific action]?"

Example actions:
- "Build the test binaries for you"
- "Compile kernels for your GPU arch"
- "Download a smaller model that fits your VRAM"
- "Check if another process is using the GPU"

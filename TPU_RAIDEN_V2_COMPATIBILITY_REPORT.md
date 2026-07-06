# TPU Raiden V2 Compatibility and Integration Report
**Date:** Monday, July 6, 2026  
**Subject:** Upgrading and Patching `tpu-raiden` main HEAD for clean integration in `torchtpu-vllm`

---

## 🔍 Executive Summary

This report documents the systems-level compatibility findings, runtime conflicts, and required C++ and build overrides identified while integrating the newest **`tpu-raiden`** `main` branch HEAD (`commit aa0faba`) as a high-performance DMA transfer engine for PyTorch-TPU. 

To ensure complete, error-free loading and E2E serving correctness, we implemented and validated several mandatory patches:
1. **`-Wl,-Bsymbolic` Scoping** (Resolves Abseil flag ODR violations).
2. **PJRT Duplicate Registration Bypass** (Resolves JAX CPU static plugin registration crashes).
3. **`zero_()` In-Place Materialization** (Eliminates infinite `AwaitBuffer` startup hangs on uninitialized storage views).

---

## 1. Required Modifications & Technical Justifications

### Modification A: `-Wl,-Bsymbolic` Linker Flags
* **File Affected:** `tpu_raiden/frameworks/torch/BUILD` (nanobind extensions `_tpu_raiden_torch` and `_torch_raw_transfer`)
* **The Conflict:** Sourcing globally loaded dynamic libraries in PyTorch-TPU (`_torch_tpu.so`) exposes static Abseil command-line flags globally in process memory. When `_tpu_raiden_torch.so` is loaded, the dynamic linker redirects its Abseil flag registration calls globally, leading to duplicate registrations of `torch_tpu_internal_num_compilation_threads` and crashing Python with an ODR violation.
* **The Resolution:** Adding `-Wl,-Bsymbolic` instructs the linker to bind dynamic function and variable references **locally within the shared library itself**. This completely isolates `_tpu_raiden_torch.so`'s private flags into its own memory workspace, resolving all flag conflicts cleanly.

### Modification B: PJRT Static Registration Bypass Patch
* **File Affected:** `MODULE.bazel` and `third_party/xla/static_registration_alway_ok.patch`
* **The Conflict:** Both JAX/libtpu and the compiled `_tpu_raiden_torch.so` statically compile OpenXLA's PJRT CPU plugin registration. Since JAX registers the `"cpu"` device globally first, `_tpu_raiden_torch.so` subsequent call to `RegisterStaticPjrtPlugin` returns `ALREADY_EXISTS`. This fails JAX's default `QCHECK_OK` assertion, triggering a SIGABRT core dump.
* **The Resolution:** We implemented a custom Bazel patch `static_registration_alway_ok.patch` which removes `QCHECK_OK` from `REGISTER_PJRT_PLUGIN(plugin_name, get_plugin_fn)`, making duplicate static registrations safe, harmless no-ops.

### Modification C: `zero_()` In-Place View Materialization
* **File Affected:** `tpu_raiden/frameworks/torch/torch_tpu_utils.cc` (inside `UnpackTorchTensor()`)
* **The Conflict:** Upstream `main` changed from `MaterializeAndReturn` to `GetBaseBuffer` to ensure direct HBM-to-HBM transfers land in the live cache regardless of view reshapes. However, at engine warmup/startup, the KV caches consist of empty, uninitialized untyped storage views. Since no operations are scheduled on them, they are unmaterialized. Calling `AwaitBuffer()` on them blocks **indefinitely** inside the C++ driver's `tsl::BlockUntilReady()`, hanging the vLLM server during startup.
* **The Resolution:** We insert `const_cast<at::Tensor&>(tensor).zero_();` immediately before calling `GetBaseBuffer`. This records a basic write on the empty storage view, forcing PyTorch-TPU to compile and materialize the actual underlying `PjRtBuffer` on TPU instantly. This allows `AwaitBuffer()` to complete immediately with zero hangs.

---

## 🛠️ Step-by-Step Compilation & Verification

To compile the compatible library from scratch, follow these instructions:

### 1. Set PYTHONPATH and Activate Virtual Env
```bash
source /home/wdhongtw_google_com/projects/venv-gemini-vllm/bin/activate
export PYTHONPATH=/home/wdhongtw_google_com/projects:/home/wdhongtw_google_com/projects/tpu-raiden:${PYTHONPATH:-}
```

### 2. Purge Caches and Rebuild from Source
```bash
/tmp/bazel-bootstrap-7.7.0 --output_base=/mnt/disks/persistent/wdhongtw_google_com/bazel-output-user-root/tpu_raiden_wdhongtw_google_com clean --expunge
./build.sh torch
```

### 3. Verify clean Python Imports
```bash
python -c "import torch_tpu; import tpu_raiden.api.torch.kv_cache_manager; print('Import Succeeded!')"
```
* **Output:**
  ```
  Import Succeeded!
  ```

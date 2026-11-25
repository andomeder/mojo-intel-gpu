# mojo_intel_gpu

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24544276.svg)](https://doi.org/10.5281/zenodo.24544276)

Intel GPU support for Mojo via Level Zero FFI.

## Overview

`mojo_intel_gpu` provides a high-level Mojo API for Intel GPU programming using the Level Zero runtime. It enables zero-overhead FFI calls to Intel's GPU compute stack, similar to how Mojo interacts with NVIDIA/AMD GPUs.

**Tested Hardware:**

- Intel Arc B580 (Battlemage Xe2) — verified working
- Intel Arc Pro B60 (Battlemage Xe2, 24 GB), reported working by a community
  tester; numbers below

**Untested (should work with Level Zero support):**

- Intel Arc A-series (Alchemist) — untested
- Intel Data Center GPUs — untested
- Other Intel GPUs with Level Zero support — untested

**Requirements:**

- Mojo v1.2.0+ required
- Level Zero runtime (`libze_loader.so`)
- Intel GPU with Level Zero support
- Linux (kernel 6.x+)

**Development Prerequisites (Arch Linux):**

```bash
# Required packages
sudo pacman -S level-zero-headers level-zero-loader intel-compute-runtime
sudo pacman -S clang llvm          # For SPIR-V compilation

# Optional tools
sudo pacman -S clinfo              # Check OpenCL device info
```

With Intel oneAPI installed, `tools/compile_kernels.sh` uses its `clang` and
`llvm-spirv` when they are not on `PATH`.

## Installation

### Option A: Install from prefix.dev (Recommended)

The package is published on the [modular-community channel](https://prefix.dev/channels/modular-community/packages/mojo-intel-gpu). Install with pixi:

```bash
pixi add mojo-intel-gpu
```

This pulls the pre-compiled `.mojoc` and pins the compatible Mojo compiler.

After installation, get the example sources and SPIR-V kernels (kernels aren't shipped in the package — they're compiled from `.cl` at build time):

```bash
git clone https://github.com/andomeder/mojo-intel-gpu
cd mojo-intel-gpu
./tools/compile_kernels.sh
mojo run -I . examples/vector_add.mojo
```

### Option B: Build from Source

#### 1. Install Level Zero Runtime

```bash
# Ubuntu/Debian
sudo apt install level-zero level-zero-dev

# Arch Linux
sudo pacman -S level-zero-loader level-zero-headers intel-compute-runtime

# Verify installation
ls -la /usr/lib/libze_loader.so
ls -la /usr/include/level_zero/ze_api.h
```

#### 2. Clone and Build

```bash
git clone https://github.com/andomeder/mojo-intel-gpu.git
cd mojo-intel-gpu

# Compile OpenCL kernels to SPIR-V
./tools/compile_kernels.sh

# Build Mojo package
mojo precompile mojo_intel_gpu -o mojo_intel_gpu.mojoc
```

#### 3. Run Examples

```bash
mojo run -I . examples/vector_add.mojo
mojo run -I . examples/matrix_multiply.mojo
mojo run -I . examples/reduction.mojo
mojo run -I . examples/benchmark.mojo
```

## Local Development with uv

For contributors editing the source, [uv](https://docs.astral.sh/uv/) handles the Python and Mojo toolchain. The `pyproject.toml` and `uv.lock` are checked in.

```bash
git clone https://github.com/andomeder/mojo-intel-gpu.git
cd mojo-intel-gpu

# Sync the environment (installs mojo from the modular PyPI index)
uv sync

# Mojo is now available on the PATH
mojo --version

# Build the package
mojo precompile mojo_intel_gpu -o mojo_intel_gpu.mojoc

# Run an example
mojo run -I . examples/vector_add.mojo
```

The `.venv/` is git-ignored. To remove and recreate: `rm -rf .venv && uv sync`.

The tree is formatter-clean. Run `mojo format mojo_intel_gpu examples tests` before opening a PR.

Note: this workflow is for editing and testing the package source. To install the published `.mojoc` in another project, use **Option A** (pixi) - Mojo packages are distributed through conda, not PyPI.

## Building the Conda Recipe

The package includes a [rattler-build](https://rattler.build) recipe at `recipe/recipe.yaml`. To rebuild from source as a conda package:

```bash
# Install rattler-build
pixi global install rattler-build

# Build the package (channels from the official Modular packaging docs)
rattler-build build \
  --recipe recipe/recipe.yaml \
  -c conda-forge \
  -c https://conda.modular.com/max \
  -c https://repo.prefix.dev/modular-community
```

The recipe installs `mojo_intel_gpu.mojoc` to `${PREFIX}/lib/mojo/` for auto-discovery by the Mojo compiler.

## Quick Start

```mojo
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount

def main() raises:
    # Initialize GPU context
    var ctx = IntelGPUContext()
    print("GPU:", ctx.device_info().name)

    # Allocate memory
    var N = 1024
    var bytes = UInt64(N * 4)
    var h_a = ctx.allocate_host(bytes)
    var h_b = ctx.allocate_host(bytes)
    var h_c = ctx.allocate_host(bytes)
    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)

    # Initialize data
    var a_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_a)
    var b_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_b)
    for i in range(N):
        a_ptr.unsafe_offset(i)[] = Float32(i)
        b_ptr.unsafe_offset(i)[] = Float32(i) * 2.0

    # Copy to device
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)

    # Load and run kernel
    var kernel = Kernel(
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add"
    )
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(N))
    var num_groups = UInt32((N + 255) / 256)
    kernel.set_group_size(256, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()

    # Copy result back
    ctx.memcpy_dtoh(h_c, d_c, bytes)

    # Verify
    var c_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_c)
    for i in range(10):
        print("c[", i, "] =", c_ptr.unsafe_offset(i)[])

    # Cleanup
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.free_host(h_a)
    ctx.free_host(h_b)
    ctx.free_host(h_c)
    ctx.close()
```

## API Reference

### Core Classes

#### `IntelGPUContext`

High-level context for Intel GPU operations.

```mojo
ctx = IntelGPUContext()  # Initialize and discover GPU

# Properties
ctx.device_info()      # DeviceProperties
ctx.compute_info()     # ComputeProperties
ctx.is_initialized()   # Bool

# Memory management
ptr = ctx.allocate_device(size)   # Allocate device memory
ptr = ctx.allocate_host(size)     # Allocate host memory
ctx.free_device(ptr)              # Free device memory
ctx.free_host(ptr)                # Free host memory

# Data transfer
ctx.memcpy_htod(dst_device, src_host, size)   # Host -> Device
ctx.memcpy_dtoh(dst_host, src_device, size)   # Device -> Host

# Synchronization
ctx.synchronize()      # Wait for GPU operations

# Cleanup
ctx.close()            # Release all resources
```

#### `Kernel`

SPIR-V kernel wrapper.

```mojo
kernel = Kernel(ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
                "path/to/kernel.spv", "kernel_name")

# Argument binding
kernel.set_arg_pointer(index, device_ptr)    # Device pointer arg
kernel.set_arg_value(index, size, value)     # Scalar value arg

# Launch configuration
kernel.set_group_size(x, y, z)               # Set work group size
kernel.launch(ZeGroupCount(x, y, z))         # Launch kernel
```

### Type Definitions

#### `DeviceProperties`

```mojo
struct DeviceProperties:
    vendor_id: UInt32
    device_id: UInt32
    device_type: ZeDeviceType      # GPU, CPU, FPGA, MCA
    core_clock_mhz: UInt32
    max_mem_alloc: UInt64
    name: String
```

#### `ComputeProperties`

```mojo
struct ComputeProperties:
    max_group_size: UInt32
    max_group_count_x: UInt32
    max_group_count_y: UInt32
    max_group_count_z: UInt32
    max_shared_local_memory: UInt32
    num_sub_group_sizes: UInt32
    sub_group_sizes: List[UInt32]
```

## Examples

### Vector Addition

```bash
mojo run -I . examples/vector_add.mojo
```

Demonstrates basic GPU operations: memory allocation, data transfer, kernel dispatch.

### Matrix Multiplication

```bash
mojo run -I . examples/matrix_multiply.mojo
```

Demonstrates 2D kernel launch with naive matrix multiplication.

### Parallel Reduction

```bash
mojo run -I . examples/reduction.mojo
```

Demonstrates shared memory and parallel tree reduction.

### Benchmark

```bash
mojo run -I . examples/benchmark.mojo
```

Measures kernel dispatch latency, memory transfer bandwidth, and end-to-end performance.

## Architecture

```
Mojo Application
    |
    v
mojo_intel_gpu (high-level API)
    |
    +-- IntelGPUContext          <- Device discovery, lifecycle
    +-- DeviceBuffer/HostBuffer  <- RAII memory management
    +-- Kernel                   <- SPIR-V loading, dispatch
    |
    v
Level Zero C API (via FFI)
    |
    +-- zeInitDrivers/zeDeviceGet  <- Enumeration
    +-- zeMemAllocDevice/zeMemFree      <- Memory
    +-- zeModuleCreate/zeKernelCreate   <- Kernels
    +-- zeCommandListAppendLaunchKernel <- Dispatch
    |
    v
libze_loader.so (Intel Compute Runtime)
    |
    v
Intel GPU Hardware
```

## Performance

FFI overhead is **zero** — resolved symbols are invoked through the C ABI as direct CALL instructions, identical to calling C from C.

**Keep setup out of loops.** Context create and close cost about 33 ms, and `Kernel()` about 90 us, all driver round trips. Build both once and reuse them; a harness that rebuilds them per iteration measures Level Zero startup, not the kernel.

**Intel Arc B580 (Battlemage Xe2) — Measured Results:**

| Metric                                 | Value                                    |
| -------------------------------------- | ---------------------------------------- |
| Device                                 | Intel Arc B580, 20 Xe-cores, 12 GB GDDR6 |
| Max Group Size                         | 1024                                     |
| Max SLM                                | 128 KB per Xe-core                       |
| GPU init (dlopen + zeInitDrivers + discovery) | 31 ms (one-time) |
| Device alloc (1 MB)                    | 0.25 us                                  |
| Host alloc (1 MB)                      | 0.19 us                                  |
| Memcpy H2D (1 MB)                      | 83.7 us avg (12.4 GB/s effective)        |
| Memcpy D2H (1 MB)                      | 107.6 us avg (9.8 GB/s effective)        |
| Kernel dispatch (set_args+launch+sync) | 6.9 us avg, 5.4 us min                   |
| E2E vector_add (1M floats)             | 1.3 ms avg, 1.0 ms min                   |

_Run `mojo run -I . examples/benchmark.mojo` to reproduce._

**Comparison with published Arc B580 benchmarks:**

- clpeak kernel latency (CR 25.31): 1.39 us (raw L0 only). Our 5.4 us includes Mojo FFI + 4 set_arg calls + launch + sync.
- SHOC PCIe H2D bandwidth: 14.5 GB/s. Our 12.4 GB/s is reasonable for 1 MB transfers via immediate command list.
- clpeak FP32: ~14,500-16,600 GFLOPS. Memory bandwidth: ~425 GB/s (93% of 456 GB/s theoretical).

**Updated bandwidth & latency (from `tests/test_bandwidth.mojo` and `tests/test_latency.mojo`):**

| Metric                          | Min         | Median      | Intel Published |
|---------------------------------|-------------|-------------|-----------------|
| H2D (1 MB, PCIe Gen4 x8)        | 12.84 GB/s  | 12.19 GB/s  | 14.5 GB/s       |
| D2H (1 MB, PCIe Gen4 x8)        | 11.11 GB/s  | 10.78 GB/s  | 14.5 GB/s       |
| Fill (1 MB, write)              | 176 GB/s    | 173 GB/s    | —               |
| D2D via host (64 MB)            | 12.15 GB/s  | 12.24 GB/s  | —               |
| vec_copy dispatch (4 MB)        | 2352 GB/s   | 1920 GB/s   | —               |
| Empty kernel launch (async)     | 1.80 µs     | 1.84 µs     | 1.39 µs         |
| Empty kernel + sync             | 5.32 µs     | 5.75 µs     | —               |
| Synchronize only (no work)      | 0.07 µs     | 0.07 µs     | —               |

_Run `mojo run -I . tests/test_bandwidth.mojo` and `tests/test_latency.mojo` to reproduce._

Note: the tests print average, minimum and median. The mean is dragged by outlier samples, most visibly the first host-to-device transfer (which pays command-list and sync setup) and the occasional scheduler hiccup on launch, so the minimum is the sustained figure and the median the typical one. Figures above are one run on an Arc B580 in November 2026; the PCIe minimums move by under 10% between runs, while fill and dispatch can move by a third. The `vec_copy` row times kernel dispatch, not execution, so its GB/s is launch rate rather than memory throughput.

### Intel Arc Pro B60 (community-reported)

From [@jporra](https://github.com/jporra) in [issue #1](https://github.com/andomeder/mojo-intel-gpu/issues/1). Measured independently, so indicative rather than a rerun: Mojo 1.0.0b3, v0.1.0, Ubuntu 26.04, Level Zero 1.28.2, Compute Runtime 1.14.37020, 24 GB per card, one device per context so the second card went unused. Examples and benchmark gave correct results.

| Metric                            | B60 (reported) | B580 (here)            |
| --------------------------------- | -------------- | ---------------------- |
| Memcpy H2D (1 MB)                 | 83.6 us        | 83.7 us avg            |
| Memcpy D2H (1 MB)                 | 100 us         | 107.6 us avg           |
| Kernel dispatch (set+launch+sync) | 7.6 us         | 6.9 us avg, 5.4 us min |
| E2E vector_add (1M floats)        | 2.89 ms        | 1.3 ms avg, 1.0 ms min |

Allocation rows are left out. They sit two orders of magnitude off the B580 figures, which points at what was timed rather than the card, and the raw output is still pending.

## Troubleshooting

### "No Intel GPU found"

Ensure your Intel GPU is detected:

```bash
# Check for Intel GPU in PCI
lspci | grep -i intel

# Check Level Zero runtime
ze_info

# Check OpenCL devices
clinfo

# Check device permissions
ls -la /dev/dri/renderD128
```

### "Module build failed"

Ensure SPIR-V is compiled correctly:

```bash
# Recompile kernels
./tools/compile_kernels.sh

# Check SPIR-V validity
spirv-val vector_add.spv
```

### "Permission denied"

Add your user to the `render` group:

```bash
sudo usermod -aG render $USER
# Log out and back in
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `mojo run -I . tests/test_all.mojo`
5. Run bandwidth/latency: `mojo run -I . tests/test_bandwidth.mojo` and `tests/test_latency.mojo`

6. Submit a pull request



## Citing this work

This project's poster has a DOI. Cite the poster when you refer to the Intel Arc
performance characterisation; cite the repository when you use the code.

> Obino, William. (2026). *Bringing Modern GPU Compute to Underrepresented Hardware:
> A Mojo/Level Zero Implementation for Intel Arc (Battlemage Xe2)* [Poster]. Poster GP-42,
> Deep Learning Indaba 2026 (Research in Africa Days), Lagos. Zenodo.
> https://doi.org/10.5281/zenodo.24544276

```bibtex
@misc{obino2026mojo,
  author       = {Obino, William},
  title        = {Bringing Modern GPU Compute to Underrepresented Hardware:
                  A Mojo/Level Zero Implementation for Intel Arc (Battlemage Xe2)},
  year         = {2026},
  howpublished = {Poster GP-42, Deep Learning Indaba 2026 (Research in Africa Days), Lagos},
  doi          = {10.5281/zenodo.24544276},
  url          = {https://doi.org/10.5281/zenodo.24544276},
  note         = {Code: https://github.com/andomeder/mojo-intel-gpu (MIT)}
}
```

Poster licence: CC BY 4.0. Code licence: MIT.

## License

MIT License — see LICENSE file.

## Acknowledgments

- Intel for the Level Zero runtime and compute-runtime (MIT license)
- Modular for Mojo language and FFI capabilities
- MLIR community for the xevm and spirv dialects

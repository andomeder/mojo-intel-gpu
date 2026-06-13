# mojo_intel_gpu

Intel GPU support for Mojo via Level Zero FFI.

## Overview

`mojo_intel_gpu` provides a high-level Mojo API for Intel GPU programming using the Level Zero runtime. It enables zero-overhead FFI calls to Intel's GPU compute stack, similar to how Mojo interacts with NVIDIA/AMD GPUs.

**Tested Hardware:**

- Intel Arc B580 (Battlemage Xe2) — verified working

**Untested (should work with Level Zero support):**

- Intel Arc A-series (Alchemist) — untested
- Intel Data Center GPUs — untested
- Other Intel GPUs with Level Zero support — untested

**Requirements:**

- Mojo v1.0.0b1+ required
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

## Installation

### 1. Install Level Zero Runtime

```bash
# Ubuntu/Debian
sudo apt install level-zero level-zero-dev

# Arch Linux
sudo pacman -S level-zero-loader level-zero-headers intel-compute-runtime

# Verify installation
ls -la /usr/lib/libze_loader.so
ls -la /usr/include/level_zero/ze_api.h
```

### 2. Clone and Build

```bash
git clone https://github.com/andomeder/mojo-intel-gpu.git
cd mojo-intel-gpu

# Compile OpenCL kernels to SPIR-V
./tools/compile_kernels.sh

# Build Mojo package
.venv/bin/mojo package mojo_intel_gpu -o mojo_intel_gpu.mojopkg
```

### 3. Run Examples

```bash
.venv/bin/mojo run -I . examples/vector_add.mojo
.venv/bin/mojo run -I . examples/matrix_multiply.mojo
.venv/bin/mojo run -I . examples/reduction.mojo
.venv/bin/mojo run -I . examples/benchmark.mojo
```

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
    var a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_a)
    var b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_b)
    for i in range(N):
        a_ptr[i] = Float32(i)
        b_ptr[i] = Float32(i) * 2.0

    # Copy to device
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)

    # Load and run kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add"
    )
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(N))
    var num_groups = UInt32((N + 255) / 256)
    kernel.set_group_size(256, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()

    # Copy result back
    ctx.memcpy_dtoh(h_c, d_c, bytes)

    # Verify
    var c_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_c)
    for i in range(10):
        print("c[", i, "] =", c_ptr[i])

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
kernel = Kernel(ctx.context(), ctx.device(), ctx.command_list(),
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
.venv/bin/mojo run -I . examples/vector_add.mojo
```

Demonstrates basic GPU operations: memory allocation, data transfer, kernel dispatch.

### Matrix Multiplication

```bash
.venv/bin/mojo run -I . examples/matrix_multiply.mojo
```

Demonstrates 2D kernel launch with naive matrix multiplication.

### Parallel Reduction

```bash
.venv/bin/mojo run -I . examples/reduction.mojo
```

Demonstrates shared memory and parallel tree reduction.

### Benchmark

```bash
.venv/bin/mojo run -I . examples/benchmark.mojo
```

Measures kernel dispatch latency, memory transfer bandwidth, and end-to-end performance.

## Project Structure

```
mojo-intel-gpu/
├── mojo_intel_gpu/            # Mojo package (importable)
│   ├── __init__.mojo          # Package entry point
│   ├── l0/
│   │   ├── types.mojo         # Level Zero type definitions
│   │   └── loader.mojo        # Level Zero function bindings (16 functions)
│   ├── core/
│   │   ├── context.mojo       # GPU context management
│   │   ├── memory.mojo        # DeviceBuffer/HostBuffer RAII wrappers
│   │   └── kernel.mojo        # SPIR-V kernel loading and dispatch
│   └── utils/
│       └── detect.mojo        # GPU detection utilities
│
├── vector_add.cl              # OpenCL kernel sources
├── matmul.cl
├── reduction.cl
│
├── examples/                  # Example programs
│   ├── vector_add.mojo
│   ├── matrix_multiply.mojo
│   ├── reduction.mojo
│   └── benchmark.mojo
│
├── tests/
│   └── test_all.mojo          # Test suite
│
├── tools/
│   └── compile_kernels.sh     # Compile OpenCL C -> SPIR-V
│
└── recipe/
    └── recipe.yaml            # Conda recipe for distribution
```

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

FFI overhead is **zero** — `lib.call["name", RetType](args)` compiles to a direct LLVM CALL instruction, identical to calling C from C.

**Intel Arc B580 (Battlemage Xe2) — Measured Results:**

| Metric                                 | Value                                    |
| -------------------------------------- | ---------------------------------------- |
| Device                                 | Intel Arc B580, 20 Xe-cores, 12 GB GDDR6 |
| Max Group Size                         | 1024                                     |
| Max SLM                                | 128 KB per Xe-core                       |
| GPU init (dlopen + zeInitDrivers + discovery) | 23.7 ms (one-time) |
| Device alloc (1 MB)                    | 0.25 us                                  |
| Host alloc (1 MB)                      | 0.19 us                                  |
| Memcpy H2D (1 MB)                      | 83.7 us avg (12.4 GB/s effective)        |
| Memcpy D2H (1 MB)                      | 107.6 us avg (9.8 GB/s effective)        |
| Kernel dispatch (set_args+launch+sync) | 6.9 us avg, 5.4 us min                   |
| E2E vector_add (1M floats)             | 1.3 ms avg, 1.0 ms min                   |

_Run `.venv/bin/mojo run -I . examples/benchmark.mojo` to reproduce._

**Comparison with published Arc B580 benchmarks:**

- clpeak kernel latency (CR 25.31): 1.39 us (raw L0 only). Our 5.4 us includes Mojo FFI + 4 set_arg calls + launch + sync.
- SHOC PCIe H2D bandwidth: 14.5 GB/s. Our 12.4 GB/s is reasonable for 1 MB transfers via immediate command list.
**Comparison with published Arc B580 benchmarks:**

- clpeak kernel latency (CR 25.31): 1.39 us (raw L0 only). Our 5.4 us includes Mojo FFI + 4 set_arg calls + launch + sync.
- SHOC PCIe H2D bandwidth: 14.5 GB/s. Our 12.4 GB/s is reasonable for 1 MB transfers via immediate command list.
- clpeak FP32: ~14,500-16,600 GFLOPS. Memory bandwidth: ~425 GB/s (93% of 456 GB/s theoretical).

**Updated bandwidth & latency (from `tests/test_bandwidth.mojo` and `tests/test_latency.mojo`):**

| Metric                          | Min         | Avg         | Intel Published |
|---------------------------------|-------------|-------------|-----------------|
| H2D (1 MB, PCIe Gen4 x8)        | 12.78 GB/s  | 1.13 GB/s   | 14.5 GB/s       |
| D2H (1 MB, PCIe Gen4 x8)        | 10.66 GB/s  | 10.09 GB/s  | 14.5 GB/s       |
| Fill (1 MB, write)              | 172 GB/s    | 43.8 GB/s   | —               |
| D2D via host (64 MB)            | 12.72 GB/s  | 10.49 GB/s  | —               |
| vec_copy kernel (4 MB)          | 1592 GB/s   | 1592 GB/s   | 425 GB/s        |
| Empty kernel launch (async)     | 2.03 µs     | 2.03 µs     | 1.39 µs         |
| Empty kernel + sync             | 4.35 µs     | 4.90 µs     | —               |
| Synchronize only (no work)      | 0.12 µs     | 0.14 µs     | —               |

_Run `.venv/bin/mojo run -I . tests/test_bandwidth.mojo` and `tests/test_latency.mojo` to reproduce._

Note: average H2D is lower than min because the first transfer includes command list + sync setup overhead. Sustained throughput (min) matches Intel's published PCIe Gen4 x8 number within 12%.


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
4. Run tests: `.venv/bin/mojo run -I . tests/test_all.mojo`
5. Run bandwidth/latency: `.venv/bin/mojo run -I . tests/test_bandwidth.mojo` and `tests/test_latency.mojo`

6. Submit a pull request



## License

MIT License — see LICENSE file.

## Acknowledgments

- Intel for the Level Zero runtime and compute-runtime (MIT license)
- Modular for Mojo language and FFI capabilities
- MLIR community for the xevm and spirv dialects

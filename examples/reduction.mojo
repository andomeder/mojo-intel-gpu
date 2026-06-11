"""Reduction example using mojo_intel_gpu package.

Demonstrates:
  - Parallel reduction with shared memory
  - Partial sum reduction (one per workgroup)
  - Block-level synchronization

Usage:
    mojo run -I . examples/reduction.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount


def main() raises:
    print("=== Parallel Reduction: Mojo → Level Zero → Intel Arc B580 ===")

    var ctx = IntelGPUContext()
    print("GPU initialized:", ctx.device_info().name)

    # Parameters
    var N = 1024 * 1024  # 1M elements
    var bytes = UInt64(N * 4)
    var group_size = UInt32(256)
    var num_groups = UInt32((N + 255) / 256)
    var output_bytes = UInt64(Int(num_groups) * 4)

    # Allocate host buffers
    var h_input = ctx.allocate_host(bytes)
    var h_output = ctx.allocate_host(output_bytes)

    # Initialize input data
    var input_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_input)
    for i in range(N):
        input_ptr[i] = 1.0  # sum should be N

    # Zero output buffer
    var output_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_output)
    for i in range(Int(num_groups)):
        output_ptr[i] = 0.0

    # Allocate device buffers
    var d_input = ctx.allocate_device(bytes)
    var d_output = ctx.allocate_device(output_bytes)
    print("Allocated device memory")

    # Copy to device
    ctx.memcpy_htod(d_input, h_input, bytes)
    ctx.memcpy_htod(d_output, h_output, output_bytes)
    print("Copied data to device")

    # Load kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./reduction.spv", "reduce_sum"
    )
    print("Kernel loaded")

    # Set kernel arguments
    kernel.set_arg_pointer(0, d_input)
    kernel.set_arg_pointer(1, d_output)
    kernel.set_arg_value(2, 4, UInt64(N))
    print("Args set")

    # Launch kernel
    print("Launching kernel (", num_groups, " groups x", group_size, " threads)...")

    kernel.set_group_size(group_size, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))

    ctx.synchronize()
    print("Kernel executed")

    # Copy result back
    ctx.memcpy_dtoh(h_output, d_output, output_bytes)

    # Verify — sum all partial results
    var total: Float32 = 0.0
    for i in range(Int(num_groups)):
        total += output_ptr[i]

    var expected = Float32(N)
    var diff = abs(total - expected)

    print("Result:", total)
    print("Expected:", expected)
    print("Difference:", diff)

    if diff < 1.0:
        print("Reduction correct!")
    else:
        print("Reduction FAILED")

    # Cleanup
    ctx.free_device(d_input)
    ctx.free_device(d_output)
    ctx.free_host(h_input)
    ctx.free_host(h_output)
    ctx.close()
    print("\nDone!")

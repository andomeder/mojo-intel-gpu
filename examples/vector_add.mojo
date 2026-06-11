"""Vector addition example using mojo_intel_gpu package.

Demonstrates basic GPU operations:
  - Device initialization
  - Memory allocation (host + device)
  - Data transfer (host → device, device → host)
  - Kernel loading and dispatch
  - Result verification

Usage:
    mojo run -I . examples/vector_add.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount


def main() raises:
    print("=== Vector Addition: Mojo → Level Zero → Intel Arc B580 ===")

    # Initialize GPU context
    var ctx = IntelGPUContext()
    print("GPU initialized:", ctx.device_info().name)

    # Parameters
    var N = 1024
    var bytes = UInt64(N * 4)  # float32 = 4 bytes

    # Allocate host buffers
    var h_a = ctx.allocate_host(bytes)
    var h_b = ctx.allocate_host(bytes)
    var h_c = ctx.allocate_host(bytes)

    # Initialize input data on host
    var a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_a)
    var b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_b)
    for i in range(N):
        a_ptr[i] = Float32(i)
        b_ptr[i] = Float32(i) * 2.0

    # Allocate device buffers
    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)
    print("Allocated device memory (", bytes * 3, " bytes)")

    # Copy host → device
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)
    print("Copied A and B to device")

    # Load kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add"
    )
    print("Kernel loaded")

    # Set kernel arguments
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(N))  # int32 size
    print("Args set")

    # Launch kernel
    var group_size = UInt32(256)
    var num_groups = UInt32((N + 255) / 256)
    print("Launching kernel (", num_groups, " groups x", group_size, " threads)...")

    kernel.set_group_size(group_size, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))

    # Wait for completion
    ctx.synchronize()
    print("Kernel executed")

    # Copy device → host
    ctx.memcpy_dtoh(h_c, d_c, bytes)

    # Verify results
    var c_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_c)
    var errors = 0
    for i in range(N):
        var expected = Float32(i) + Float32(i) * 2.0
        var actual = c_ptr[i]
        if actual != expected:
            if errors < 5:
                print("  MISMATCH [", i, "]: expected", expected, "got", actual)
            errors += 1

    if errors == 0:
        print("ALL", N, "results correct! c[i] = a[i] + b[i] = i + 2i = 3i")
        print("  c[0] =", c_ptr[0], " (expect 0)")
        print("  c[1] =", c_ptr[1], " (expect 3)")
        print("  c[99] =", c_ptr[99], " (expect 297)")
    else:
        print(errors, "mismatches out of", N)

    # Cleanup
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.free_host(h_a)
    ctx.free_host(h_b)
    ctx.free_host(h_c)
    ctx.close()
    print("\nDone!")

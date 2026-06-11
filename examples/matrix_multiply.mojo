"""Matrix multiplication example using mojo_intel_gpu package.

Demonstrates:
  - 2D kernel launch with naive matrix multiplication
  - Shared memory usage via SLM
  - Performance measurement

Usage:
    mojo run -I . examples/matrix_multiply.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount


def main() raises:
    print("=== Matrix Multiplication: Mojo → Level Zero → Intel Arc B580 ===")

    var ctx = IntelGPUContext()
    print("GPU initialized:", ctx.device_info().name)

    # Parameters — 64x64 matrices
    var M = 64
    var N = 64
    var K = 64
    var bytes_a = UInt64(M * K * 4)
    var bytes_b = UInt64(K * N * 4)
    var bytes_c = UInt64(M * N * 4)

    # Allocate host buffers
    var h_a = ctx.allocate_host(bytes_a)
    var h_b = ctx.allocate_host(bytes_b)
    var h_c = ctx.allocate_host(bytes_c)

    # Initialize matrices
    var a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_a)
    var b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_b)
    for i in range(M):
        for j in range(K):
            a_ptr[i * K + j] = Float32(i + j) * 0.01
    for i in range(K):
        for j in range(N):
            b_ptr[i * N + j] = Float32(i - j) * 0.01

    # Allocate device buffers
    var d_a = ctx.allocate_device(bytes_a)
    var d_b = ctx.allocate_device(bytes_b)
    var d_c = ctx.allocate_device(bytes_c)
    print("Allocated device memory")

    # Copy to device
    ctx.memcpy_htod(d_a, h_a, bytes_a)
    ctx.memcpy_htod(d_b, h_b, bytes_b)
    print("Copied matrices to device")

    # Load kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./matmul.spv", "matmul"
    )
    print("Kernel loaded")

    # Set kernel arguments
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(M))
    kernel.set_arg_value(4, 4, UInt64(N))
    kernel.set_arg_value(5, 4, UInt64(K))
    print("Args set")

    # Launch kernel — 16x16 tile, so 4x4 groups for 64x64
    var TILE = 16
    var group_x = UInt32((N + TILE - 1) / TILE)
    var group_y = UInt32((M + TILE - 1) / TILE)
    print("Launching kernel (", group_x, "x", group_y, " groups of 16x16)...")

    kernel.set_group_size(UInt32(TILE), UInt32(TILE), 1)
    kernel.launch(ZeGroupCount(group_x, group_y, 1))

    ctx.synchronize()
    print("Kernel executed")

    # Copy result back
    ctx.memcpy_dtoh(h_c, d_c, bytes_c)

    # Verify — compute CPU reference and compare
    var c_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_c)
    var errors = 0
    var max_diff: Float32 = 0.0

    for i in range(M):
        for j in range(N):
            var sum: Float32 = 0.0
            for k in range(K):
                sum += a_ptr[i * K + k] * b_ptr[k * N + j]
            var diff = abs(c_ptr[i * N + j] - sum)
            if diff > max_diff:
                max_diff = diff
            if diff > 0.001:
                errors += 1

    print("Max difference:", max_diff)
    if errors == 0:
        print("ALL", M * N, "results correct!")
    else:
        print(errors, "mismatches out of", M * N)

    # Cleanup
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.free_host(h_a)
    ctx.free_host(h_b)
    ctx.free_host(h_c)
    ctx.close()
    print("\nDone!")

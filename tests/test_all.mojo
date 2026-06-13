"""Test suite for mojo_intel_gpu package.

Tests all major functionality:
  - Device detection
  - Context initialization
  - Memory allocation and transfers
  - Kernel loading and dispatch
  - Multi-GPU enumeration

Usage:
    mojo run -I . tests/test_all.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount
from mojo_intel_gpu.utils.detect import IntelGPUDetector


def test_detection() raises:
    """Test GPU detection utilities."""
    print("=== Test: GPU Detection ===")

    var detector = IntelGPUDetector()

    if detector.has_gpu():
        print("  ✓ Intel GPU detected")
        var count = detector.get_gpu_count()
        print("  ✓ GPU count:", count)
        var info = detector.get_gpu_info()
        print("  ✓ Device:", info.name)
        print("  ✓ Vendor:", hex(Int(info.vendor_id)))
        print("  ✓ Device ID:", hex(Int(info.device_id)))
    else:
        print("  ✗ No Intel GPU detected")
        raise Error("No Intel GPU available for testing")


def test_context() raises:
    """Test context initialization."""
    print("\n=== Test: Context Initialization ===")

    var ctx = IntelGPUContext()

    print("  ✓ Context created")
    print("  ✓ Device:", ctx.device_info().name)
    print("  ✓ Max group size:", ctx.compute_info().max_group_size)
    print("  ✓ Max SLM:", ctx.compute_info().max_shared_local_memory, "bytes")

    ctx.close()
    print("  ✓ Context closed")


def test_memory() raises:
    """Test memory allocation and transfers."""
    print("\n=== Test: Memory Management ===")

    var ctx = IntelGPUContext()

    # Test device allocation
    var d_buf = ctx.allocate_device(1024)
    print("  ✓ Device allocation (1KB)")
    ctx.free_device(d_buf)
    print("  ✓ Device free")

    # Test host allocation
    var h_buf = ctx.allocate_host(1024)
    print("  ✓ Host allocation (1KB)")
    ctx.free_host(h_buf)
    print("  ✓ Host free")

    # Test host → device → host round-trip
    var N = 256
    var bytes = UInt64(N * 4)

    var h_src = ctx.allocate_host(bytes)
    var h_dst = ctx.allocate_host(bytes)
    var d_buf2 = ctx.allocate_device(bytes)

    # Initialize source
    var src_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_src)
    for i in range(N):
        src_ptr[i] = Float32(i) * 3.14

    # Copy host → device → host
    ctx.memcpy_htod(d_buf2, h_src, bytes)
    ctx.memcpy_dtoh(h_dst, d_buf2, bytes)

    # Verify
    var dst_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_dst)
    var errors = 0
    for i in range(N):
        if dst_ptr[i] != src_ptr[i]:
            errors += 1

    if errors == 0:
        print("  ✓ Memory round-trip correct (", N, " floats)")
    else:
        print("  ✗ Memory round-trip failed:", errors, "errors")

    ctx.free_device(d_buf2)
    ctx.free_host(h_src)
    ctx.free_host(h_dst)
    ctx.close()


def test_kernel() raises:
    """Test kernel loading and dispatch."""
    print("\n=== Test: Kernel Dispatch ===")

    var ctx = IntelGPUContext()

    # Allocate buffers
    var N = 1024
    var bytes = UInt64(N * 4)

    var h_a = ctx.allocate_host(bytes)
    var h_b = ctx.allocate_host(bytes)
    var h_c = ctx.allocate_host(bytes)

    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)

    # Initialize
    var a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_a)
    var b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_b)
    for i in range(N):
        a_ptr[i] = Float32(i)
        b_ptr[i] = Float32(i) * 2.0

    # Copy to device
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)

    # Load kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add"
    )
    print("  ✓ Kernel loaded")

    # Set args
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(N))
    print("  ✓ Arguments set")

    # Launch
    var group_size = UInt32(256)
    var num_groups = UInt32((N + 255) / 256)
    kernel.set_group_size(group_size, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    print("  ✓ Kernel launched")

    ctx.synchronize()
    print("  ✓ Synchronized")

    # Verify
    ctx.memcpy_dtoh(h_c, d_c, bytes)
    var c_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h_c)
    var errors = 0
    for i in range(N):
        var expected = Float32(i) + Float32(i) * 2.0
        if c_ptr[i] != expected:
            errors += 1

    if errors == 0:
        print("  ✓ Kernel results correct (", N, " elements)")
    else:
        print("  ✗ Kernel results failed:", errors, "errors")

    # Cleanup
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.free_host(h_a)
    ctx.free_host(h_b)
    ctx.free_host(h_c)
    ctx.close()


def test_multi_gpu() raises:
    """Test multi-GPU enumeration (validates zeInitDrivers refactor)."""
    print("\n=== Test: Multi-GPU Enumeration ===")

    var detector = IntelGPUDetector()
    var count = detector.get_gpu_count()
    print("  Intel GPUs found:", count)

    if count == 0:
        raise Error("No Intel GPUs found")

    for i in range(count):
        var info = detector.get_gpu_info()
        print(
            "  GPU[", i, "]:", info.name,
            "vendor=", hex(Int(info.vendor_id)),
            "device=", hex(Int(info.device_id)),
        )

    if count >= 1:
        print("  ✓ Enumeration correct")
    else:
        raise Error("Expected at least 1 GPU")


def main() raises:
    print("=== mojo_intel_gpu Test Suite ===\n")

    var tests_passed = 0
    var tests_failed = 0

    try:
        test_detection()
        tests_passed += 1
    except e:
        print("  ✗ Detection test failed:", e)
        tests_failed += 1

    try:
        test_context()
        tests_passed += 1
    except e:
        print("  ✗ Context test failed:", e)
        tests_failed += 1

    try:
        test_memory()
        tests_passed += 1
    except e:
        print("  ✗ Memory test failed:", e)
        tests_failed += 1

    try:
        test_kernel()
        tests_passed += 1
    except e:
        print("  ✗ Kernel test failed:", e)
        tests_failed += 1

    try:
        test_multi_gpu()
        tests_passed += 1
    except e:
        print("  ✗ Multi-GPU test failed:", e)
        tests_failed += 1

    print("\n=== Test Results ===")
    print("  Passed:", tests_passed)
    print("  Failed:", tests_failed)
    print("  Total:", tests_passed + tests_failed)

    if tests_failed > 0:
        print("\n  ✗ SOME TESTS FAILED")
        raise Error("Test failures detected")
    else:
        print("\n  ✓ ALL TESTS PASSED")

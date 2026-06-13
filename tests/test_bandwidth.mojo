"""Bandwidth tests for mojo_intel_gpu package.

Measures memory transfer throughput between host and device, and device-side
fill speed. Uses L0's native memcpy + the new memset_device wrapper.

Usage:
    mojo run -I . tests/test_bandwidth.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount
from std.time.time import perf_counter_ns


# ── helpers ──────────────────────────────────────────────────────────────


def pad(s: String, width: Int) -> String:
    var result = s
    while result.byte_length() < width:
        result += " "
    return result


def print_header():
    print("")
    print("=== Intel GPU Bandwidth Test ===")
    print(
        "Test              | Size (MB)  | Avg (us)   | Min (us)   |"
        " Avg GB/s  | Min GB/s"
    )
    print(
        "------------------|------------|------------|------------|"
        "-----------|---------"
    )


def print_row(
    name: String,
    size_mb: Float64,
    avg_us: Float64,
    min_us: Float64,
    avg_gbs: Float64,
    min_gbs: Float64,
):
    print(
        pad(name, 17), " | ",
        pad(String(size_mb), 8), "    | ",
        pad(String(avg_us), 8), "   | ",
        pad(String(min_us), 8), "   | ",
        pad(String(avg_gbs), 7), "  | ",
        pad(String(min_gbs), 5),
        sep="",
    )


def ns_to_us(ns: UInt64) -> Float64:
    return Float64(ns) / 1000.0


# ── tests ────────────────────────────────────────────────────────────────


def test_h2d_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, h_buf: Int,
                       size: UInt64, iters: Int) raises:
    var size_f = Float64(size) / (1024.0 * 1024.0)
    var bytes_f = Float64(size)
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_htod(d_buf, h_buf, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
    var avg_us = ns_to_us(total) / Float64(iters)
    var min_us = ns_to_us(mn)
    var avg_gbs = bytes_f / (avg_us * 1000.0)
    var min_gbs = bytes_f / (min_us * 1000.0)
    print_row("H2D (PCIe)", size_f, avg_us, min_us, avg_gbs, min_gbs)


def test_d2h_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, h_buf: Int,
                       size: UInt64, iters: Int) raises:
    var size_f = Float64(size) / (1024.0 * 1024.0)
    var bytes_f = Float64(size)
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_dtoh(h_buf, d_buf, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
    var avg_us = ns_to_us(total) / Float64(iters)
    var min_us = ns_to_us(mn)
    var avg_gbs = bytes_f / (avg_us * 1000.0)
    var min_gbs = bytes_f / (min_us * 1000.0)
    print_row("D2H (PCIe)", size_f, avg_us, min_us, avg_gbs, min_gbs)


def test_d2d_bandwidth(mut ctx: IntelGPUContext, d_src: Int, d_dst: Int,
                       size: UInt64, iters: Int) raises:
    var size_f = Float64(size) / (1024.0 * 1024.0)
    var bytes_f = Float64(size)
    var h_temp = ctx.allocate_host(size)
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_dtoh(h_temp, d_src, size)
        ctx.memcpy_htod(d_dst, h_temp, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    ctx.free_host(h_temp)
    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
    var avg_us = ns_to_us(total) / Float64(iters)
    var min_us = ns_to_us(mn)
    # 2*size worth of PCIe traffic
    var effective = bytes_f * 2.0
    var avg_gbs = effective / (avg_us * 1000.0)
    var min_gbs = effective / (min_us * 1000.0)
    print_row("D2D (via H)", size_f, avg_us, min_us, avg_gbs, min_gbs)


def test_fill_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, size: UInt64,
                        iters: Int) raises:
    var size_f = Float64(size) / (1024.0 * 1024.0)
    var bytes_f = Float64(size)
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memset_device(d_buf, UInt8(0), size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
    var avg_us = ns_to_us(total) / Float64(iters)
    var min_us = ns_to_us(mn)
    var avg_gbs = bytes_f / (avg_us * 1000.0)
    var min_gbs = bytes_f / (min_us * 1000.0)
    print_row("Fill (write)", size_f, avg_us, min_us, avg_gbs, min_gbs)


def test_kernel_copy(mut ctx: IntelGPUContext, mut kernel: Kernel,
                     d_src: Int, d_dst: Int, n: Int, iters: Int) raises:
    var size_f = Float64(n * 4) / (1024.0 * 1024.0)
    var bytes_f = Float64(n * 4)
    var group_size = UInt32(256)
    var num_groups = UInt32((n + 255) / 256)

    kernel.set_arg_pointer(0, d_src)
    kernel.set_arg_pointer(1, d_dst)
    kernel.set_arg_value(2, 4, UInt64(n))
    kernel.set_group_size(group_size, 1, 1)

    # Warmup
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()

    # Time kernel launch (no sync per launch)
    var t0 = perf_counter_ns()
    for _ in range(iters):
        kernel.launch(ZeGroupCount(num_groups, 1, 1))
    var t1 = perf_counter_ns()
    ctx.synchronize()

    var total_ns = UInt64(t1 - t0)
    var avg_us = ns_to_us(total_ns) / Float64(iters)
    var avg_gbs = bytes_f / (avg_us * 1000.0)
    print_row("vec_copy kernel", size_f, avg_us, avg_us, avg_gbs, avg_gbs)


# ── main ─────────────────────────────────────────────────────────────────


def main() raises:
    print_header()

    var ctx = IntelGPUContext()
    var iters = 10

    # H2D/D2H/Fill: 1 MB
    var small_size: UInt64 = 1 * 1024 * 1024
    var h_buf = ctx.allocate_host(small_size)
    var d_buf = ctx.allocate_device(small_size)

    # Initialize host buffer
    var h_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=h_buf)
    for i in range(Int(small_size)):
        h_ptr[i] = UInt8(i & 0xFF)

    test_h2d_bandwidth(ctx, d_buf, h_buf, small_size, iters)
    test_d2h_bandwidth(ctx, d_buf, h_buf, small_size, iters)
    test_fill_bandwidth(ctx, d_buf, small_size, iters)

    # D2D: 64 MB
    var large_size: UInt64 = 64 * 1024 * 1024
    var d_src = ctx.allocate_device(large_size)
    var d_dst = ctx.allocate_device(large_size)
    test_d2d_bandwidth(ctx, d_src, d_dst, large_size, iters)

    # Kernel copy: 1M floats
    var n: Int = 1024 * 1024
    var kbytes: UInt64 = UInt64(n * 4)
    var d_ksrc = ctx.allocate_device(kbytes)
    var d_kdst = ctx.allocate_device(kbytes)

    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./bandwidth.spv", "vec_copy"
    )
    test_kernel_copy(ctx, kernel, d_ksrc, d_kdst, n, iters)

    # Cleanup
    ctx.free_host(h_buf)
    ctx.free_device(d_buf)
    ctx.free_device(d_src)
    ctx.free_device(d_dst)
    ctx.free_device(d_ksrc)
    ctx.free_device(d_kdst)
    ctx.close()

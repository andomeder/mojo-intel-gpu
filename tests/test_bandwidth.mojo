"""Bandwidth tests for mojo_intel_gpu package.

Measures memory transfer throughput between host and device, and device-side
fill speed. Uses L0's native memcpy and the memset_device wrapper.

Usage:
    mojo run -I . tests/test_bandwidth.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount
from std.time.time import perf_counter_ns


# ── helpers ──────────────────────────────────────────────────────────────


@fieldwise_init
struct Timings(Copyable, Movable):
    """Summary of one measurement row, in microseconds."""
    var avg_us: Float64
    var min_us: Float64
    var med_us: Float64
    var max_us: Float64


def ns_to_us(ns: UInt64) -> Float64:
    return Float64(ns) / 1000.0


def summarise(times: List[UInt64]) -> Timings:
    """Summarise nanosecond samples, with the median alongside the mean."""
    var ordered = List[UInt64]()
    for t in times:
        ordered.append(t)
    sort(ordered)
    var n = len(ordered)
    var total: UInt64 = 0
    for t in ordered:
        total += t
    var mid = n // 2
    var median_ns = Float64(ordered[mid])
    if n % 2 == 0:
        median_ns = (Float64(ordered[mid - 1]) + Float64(ordered[mid])) / 2.0
    return Timings(
        ns_to_us(total) / Float64(n),
        ns_to_us(ordered[0]),
        median_ns / 1000.0,
        ns_to_us(ordered[n - 1]),
    )


def to_mb(size: UInt64) -> Float64:
    return Float64(size) / (1024.0 * 1024.0)


def gbs(size_bytes: Float64, us: Float64) -> Float64:
    return size_bytes / (us * 1000.0)


def pad(s: String, width: Int) -> String:
    var result = s
    while result.byte_length() < width:
        result += " "
    return result


def print_header():
    print("")
    print("=== Intel GPU Bandwidth Test ===")
    print(
        "Test              | Size (MB) | Avg (us)  | Min (us)  | Med (us)  |"
        " Avg GB/s | Min GB/s"
    )
    print(
        "------------------|-----------|-----------|-----------|-----------|"
        "----------|---------"
    )


def print_row(name: String, size: UInt64, stats: Timings, size_bytes: Float64):
    print(
        pad(name, 17), " | ",
        pad(String(to_mb(size)), 9), " | ",
        pad(String(stats.avg_us), 9), " | ",
        pad(String(stats.min_us), 9), " | ",
        pad(String(stats.med_us), 9), " | ",
        pad(String(gbs(size_bytes, stats.avg_us)), 8), " | ",
        pad(String(gbs(size_bytes, stats.min_us)), 8),
        sep="",
    )


# ── tests ────────────────────────────────────────────────────────────────


def test_h2d_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, h_buf: Int,
                       size: UInt64, iters: Int) raises:
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_htod(d_buf, h_buf, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    print_row("H2D (PCIe)", size, summarise(times), Float64(size))


def test_d2h_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, h_buf: Int,
                       size: UInt64, iters: Int) raises:
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_dtoh(h_buf, d_buf, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    print_row("D2H (PCIe)", size, summarise(times), Float64(size))


def test_d2d_bandwidth(mut ctx: IntelGPUContext, d_src: Int, d_dst: Int,
                       size: UInt64, iters: Int) raises:
    var h_temp = ctx.allocate_host(size)
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_dtoh(h_temp, d_src, size)
        ctx.memcpy_htod(d_dst, h_temp, size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    ctx.free_host(h_temp)
    print_row("D2D (via H)", size, summarise(times), Float64(size) * 2.0)


def test_fill_bandwidth(mut ctx: IntelGPUContext, d_buf: Int, size: UInt64,
                        iters: Int) raises:
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memset_device(d_buf, UInt8(0), size)
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    print_row("Fill (write)", size, summarise(times), Float64(size))


def test_kernel_copy(mut ctx: IntelGPUContext, mut kernel: Kernel,
                     d_src: Int, d_dst: Int, n: Int, iters: Int) raises:
    var size = UInt64(n * 4)
    var group_size = UInt32(256)
    var num_groups = UInt32((n + 255) / 256)

    kernel.set_arg_pointer(0, d_src)
    kernel.set_arg_pointer(1, d_dst)
    kernel.set_arg_value(2, 4, UInt64(n))
    kernel.set_group_size(group_size, 1, 1)

    # Warmup
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        kernel.launch(ZeGroupCount(num_groups, 1, 1))
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    ctx.synchronize()
    print_row("vec_copy dispatch", size, summarise(times), Float64(size))


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
    var h_ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=h_buf)
    for i in range(Int(small_size)):
        h_ptr.unsafe_offset(i)[] = UInt8(i & 0xFF)

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
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
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

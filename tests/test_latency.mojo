"""Latency tests for mojo_intel_gpu package.

Measures kernel launch overhead and synchronization cost. Uses an empty
SPIR-V kernel to isolate the cost of the L0 dispatch path from any real
compute or memory work.

Usage:
    mojo run -I . tests/test_latency.mojo
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


def pad(s: String, width: Int) -> String:
    var result = s
    while result.byte_length() < width:
        result += " "
    return result


def print_header():
    print("")
    print("=== Intel GPU Latency Test ===")
    print(
        "Test                          | Iters | Avg (us)  | Min (us)  |"
        " Med (us)  | Max (us)"
    )
    print(
        "------------------------------|-------|-----------|-----------|"
        "-----------|---------"
    )


def print_row(name: String, iters: Int, stats: Timings):
    print(
        pad(name, 29), " | ",
        pad(String(iters), 4), "  | ",
        pad(String(stats.avg_us), 9), " | ",
        pad(String(stats.min_us), 9), " | ",
        pad(String(stats.med_us), 9), " | ",
        pad(String(stats.max_us), 9),
        sep="",
    )


# ── tests ────────────────────────────────────────────────────────────────


def test_empty_kernel_launch(mut ctx: IntelGPUContext, mut kernel: Kernel,
                             iters: Int) raises:
    """Measure pure launch overhead (no per-launch sync)."""
    # Warmup
    kernel.launch(ZeGroupCount(UInt32(1), 1, 1))
    ctx.synchronize()

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        kernel.launch(ZeGroupCount(UInt32(1), 1, 1))
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    ctx.synchronize()
    print_row("Empty kernel launch", iters, summarise(times))


def test_empty_kernel_sync(mut ctx: IntelGPUContext, mut kernel: Kernel,
                           iters: Int) raises:
    """Measure launch + sync (what users actually experience)."""
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        kernel.launch(ZeGroupCount(UInt32(1), 1, 1))
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    print_row("Empty kernel + sync", iters, summarise(times))


def test_sync_overhead(mut ctx: IntelGPUContext, iters: Int) raises:
    """Measure sync overhead alone (no kernel work)."""
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
    print_row("Sync only (no work)", iters, summarise(times))


# ── main ─────────────────────────────────────────────────────────────────


def main() raises:
    print_header()

    var ctx = IntelGPUContext()
    var iters = 100

    # Load empty kernel
    var kernel = Kernel(
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
        "./empty.spv", "empty"
    )
    kernel.set_group_size(UInt32(1), 1, 1)

    test_empty_kernel_launch(ctx, kernel, iters)
    test_empty_kernel_sync(ctx, kernel, iters)
    test_sync_overhead(ctx, iters)

    ctx.close()

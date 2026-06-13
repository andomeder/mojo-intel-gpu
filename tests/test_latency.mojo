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


def pad(s: String, width: Int) -> String:
    var result = s
    while result.byte_length() < width:
        result += " "
    return result


def print_header():
    print("")
    print("=== Intel GPU Latency Test ===")
    print("Test                          | Iters | Avg (us)   | Min (us)   | Max (us)")
    print("------------------------------|-------|------------|------------|---------")


def print_row(
    name: String,
    iters: Int,
    avg_us: Float64,
    min_us: Float64,
    max_us: Float64,
):
    print(
        pad(name, 29), " | ",
        pad(String(iters), 4), "  | ",
        pad(String(avg_us), 8), "   | ",
        pad(String(min_us), 8), "   | ",
        pad(String(max_us), 6),
        sep="",
    )


def ns_to_us(ns: UInt64) -> Float64:
    return Float64(ns) / 1000.0


# ── tests ────────────────────────────────────────────────────────────────


def test_empty_kernel_launch(mut ctx: IntelGPUContext, mut kernel: Kernel,
                             iters: Int) raises:
    """Measure pure launch overhead (no per-launch sync)."""
    # Warmup
    kernel.launch(ZeGroupCount(UInt32(1), 1, 1))
    ctx.synchronize()

    # Time many launches, one sync at end
    var t0 = perf_counter_ns()
    for _ in range(iters):
        kernel.launch(ZeGroupCount(UInt32(1), 1, 1))
    var t1 = perf_counter_ns()
    ctx.synchronize()

    var total_ns = UInt64(t1 - t0)  # launches only, no sync
    var avg = ns_to_us(total_ns) / Float64(iters)
    print_row("Empty kernel launch", iters, avg, avg, avg)


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

    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    var mx: UInt64 = 0
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    var avg = ns_to_us(total) / Float64(iters)
    print_row("Empty kernel + sync", iters, avg, ns_to_us(mn), ns_to_us(mx))


def test_sync_overhead(mut ctx: IntelGPUContext, iters: Int) raises:
    """Measure sync overhead alone (no kernel work)."""
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))

    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    var mx: UInt64 = 0
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    var avg = ns_to_us(total) / Float64(iters)
    print_row("Sync only (no work)", iters, avg, ns_to_us(mn), ns_to_us(mx))


# ── main ─────────────────────────────────────────────────────────────────


def main() raises:
    print_header()

    var ctx = IntelGPUContext()
    var iters = 100

    # Load empty kernel
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./empty.spv", "empty"
    )
    kernel.set_group_size(UInt32(1), 1, 1)

    test_empty_kernel_launch(ctx, kernel, iters)
    test_empty_kernel_sync(ctx, kernel, iters)
    test_sync_overhead(ctx, iters)

    ctx.close()

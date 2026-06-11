"""Benchmark script for Intel Arc B580 via Level Zero.

Measures kernel dispatch latency, memory throughput, and end-to-end
vector_add performance using the mojo_intel_gpu package.

Usage:
    mojo run -I . examples/benchmark.mojo
"""
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount
from std.time.time import perf_counter_ns


# ── helpers ──────────────────────────────────────────────────────────────


def print_header():
    print("")
    print("=== Intel Arc B580 Benchmark ===")
    print(
        "Operation              | Avg (us)  | Min (us)  | Max (us)  |"
        " Iterations"
    )
    print(
        "-----------------------|-----------|-----------|-----------|"
        "----------"
    )


def print_row(
    name: String,
    avg_us: Float64,
    min_us: Float64,
    max_us: Float64,
    iters: Int,
):
    # Pad name to 22 chars
    var padded = name
    while padded.byte_length() < 22:
        padded += " "
    print(
        padded, " | ",
        avg_us, " | ",
        min_us, " | ",
        max_us, " | ",
        iters,
        sep="",
    )


def ns_to_us(ns: UInt64) -> Float64:
    return Float64(ns) / 1000.0


# ── benchmarks ───────────────────────────────────────────────────────────


def bench_init() raises:
    """Time GPU context initialization."""
    var iters = 3
    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var ctx = IntelGPUContext()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))
        ctx.close()

    var total: UInt64 = 0
    var mn: UInt64 = UInt64.MAX
    var mx: UInt64 = 0
    for t in range(iters):
        total += times[t]
        if times[t] < mn:
            mn = times[t]
        if times[t] > mx:
            mx = times[t]
    var avg = total / UInt64(iters)
    print_row("GPU init", ns_to_us(avg), ns_to_us(mn), ns_to_us(mx), iters)


def bench_alloc_device(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time device memory allocation (1 MB)."""
    var size = UInt64(1024 * 1024)
    # warmup
    var w = ctx.allocate_device(size)
    ctx.free_device(w)

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var ptr = ctx.allocate_device(size)
        var t1 = perf_counter_ns()
        ctx.free_device(ptr)
        times.append(UInt64(t1 - t0))

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "Device alloc (1MB)",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


def bench_alloc_host(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time host memory allocation (1 MB)."""
    var size = UInt64(1024 * 1024)
    # warmup
    var w = ctx.allocate_host(size)
    ctx.free_host(w)

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        var ptr = ctx.allocate_host(size)
        var t1 = perf_counter_ns()
        ctx.free_host(ptr)
        times.append(UInt64(t1 - t0))

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "Host alloc (1MB)",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


def bench_memcpy_htod(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time host-to-device memcpy (1 MB)."""
    var size = UInt64(1024 * 1024)
    var h = ctx.allocate_host(size)
    var d = ctx.allocate_device(size)

    # warmup
    ctx.memcpy_htod(d, h, size)
    ctx.synchronize()

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_htod(d, h, size)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))

    ctx.free_device(d)
    ctx.free_host(h)

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "Memcpy H2D (1MB)",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


def bench_memcpy_dtoh(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time device-to-host memcpy (1 MB)."""
    var size = UInt64(1024 * 1024)
    var h = ctx.allocate_host(size)
    var d = ctx.allocate_device(size)

    # warmup
    ctx.memcpy_dtoh(h, d, size)
    ctx.synchronize()

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.memcpy_dtoh(h, d, size)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))

    ctx.free_device(d)
    ctx.free_host(h)

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "Memcpy D2H (1MB)",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


def bench_kernel_dispatch(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time kernel dispatch (set_args + launch + sync) for vector_add N=1024."""
    var N = 1024
    var bytes = UInt64(N * 4)
    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)

    # Load kernel once
    var kernel = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add",
    )

    var group_size = UInt32(256)
    var num_groups = UInt32((N + 255) / 256)

    # warmup
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_value(3, 4, UInt64(N))
    kernel.set_group_size(group_size, 1, 1)
    kernel.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        kernel.set_arg_pointer(0, d_a)
        kernel.set_arg_pointer(1, d_b)
        kernel.set_arg_pointer(2, d_c)
        kernel.set_arg_value(3, 4, UInt64(N))
        kernel.set_group_size(group_size, 1, 1)
        kernel.launch(ZeGroupCount(num_groups, 1, 1))
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))

    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "Kernel dispatch",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


def bench_e2e_vector_add(mut ctx: IntelGPUContext, iters: Int) raises:
    """Time end-to-end vector_add (alloc + h2d + launch + sync + dtoh) for 1M floats."""
    var N = 1_000_000
    var bytes = UInt64(N * 4)
    var group_size = UInt32(256)
    var num_groups = UInt32((N + 255) / 256)

    # warmup
    var w_a = ctx.allocate_device(bytes)
    var w_b = ctx.allocate_device(bytes)
    var w_c = ctx.allocate_device(bytes)
    var wh_a = ctx.allocate_host(bytes)
    var wh_b = ctx.allocate_host(bytes)
    var wh_c = ctx.allocate_host(bytes)
    ctx.memcpy_htod(w_a, wh_a, bytes)
    ctx.memcpy_htod(w_b, wh_b, bytes)
    var wkr = Kernel(
        ctx.context(), ctx.device(), ctx.command_list(),
        "./vector_add.spv", "vector_add",
    )
    wkr.set_arg_pointer(0, w_a)
    wkr.set_arg_pointer(1, w_b)
    wkr.set_arg_pointer(2, w_c)
    wkr.set_arg_value(3, 4, UInt64(N))
    wkr.set_group_size(group_size, 1, 1)
    wkr.launch(ZeGroupCount(num_groups, 1, 1))
    ctx.synchronize()
    ctx.memcpy_dtoh(wh_c, w_c, bytes)
    ctx.free_device(w_a)
    ctx.free_device(w_b)
    ctx.free_device(w_c)
    ctx.free_host(wh_a)
    ctx.free_host(wh_b)
    ctx.free_host(wh_c)

    var times = List[UInt64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()

        # Allocate
        var da = ctx.allocate_device(bytes)
        var db = ctx.allocate_device(bytes)
        var dc = ctx.allocate_device(bytes)
        var ha = ctx.allocate_host(bytes)
        var hb = ctx.allocate_host(bytes)
        var hc = ctx.allocate_host(bytes)

        # Copy to device
        ctx.memcpy_htod(da, ha, bytes)
        ctx.memcpy_htod(db, hb, bytes)

        # Launch
        var kr = Kernel(
            ctx.context(), ctx.device(), ctx.command_list(),
            "./vector_add.spv", "vector_add",
        )
        kr.set_arg_pointer(0, da)
        kr.set_arg_pointer(1, db)
        kr.set_arg_pointer(2, dc)
        kr.set_arg_value(3, 4, UInt64(N))
        kr.set_group_size(group_size, 1, 1)
        kr.launch(ZeGroupCount(num_groups, 1, 1))
        ctx.synchronize()

        # Copy back
        ctx.memcpy_dtoh(hc, dc, bytes)

        var t1 = perf_counter_ns()
        times.append(UInt64(t1 - t0))

        ctx.free_device(da)
        ctx.free_device(db)
        ctx.free_device(dc)
        ctx.free_host(ha)
        ctx.free_host(hb)
        ctx.free_host(hc)

    var total = UInt64(0)
    var mn = times[0]
    var mx = times[0]
    for i in range(iters):
        total += times[i]
        if times[i] < mn:
            mn = times[i]
        if times[i] > mx:
            mx = times[i]
    print_row(
        "E2E vector_add (1M)",
        ns_to_us(total / UInt64(iters)),
        ns_to_us(mn),
        ns_to_us(mx),
        iters,
    )


# ── main ─────────────────────────────────────────────────────────────────


def main() raises:
    var info = IntelGPUContext()
    print("Device:", info.device_info().name)
    info.close()

    print_header()
    bench_init()

    # Longer-running benchmarks share a single context
    var ctx = IntelGPUContext()
    bench_alloc_device(ctx, 100)
    bench_alloc_host(ctx, 100)
    bench_memcpy_htod(ctx, 100)
    bench_memcpy_dtoh(ctx, 100)
    bench_kernel_dispatch(ctx, 100)
    bench_e2e_vector_add(ctx, 10)
    ctx.close()

    print("")
    print("Benchmark complete.")

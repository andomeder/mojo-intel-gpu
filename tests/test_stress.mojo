from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount
from std.time.time import perf_counter_ns

def ns_to_ms(ns: Int) -> Float64:
    return Float64(ns) / 1_000_000.0

def main() raises:
    print("=== Load/Stress Test ===")

    # 1. Repeated init/destroy cycles (detects dlopen/context leaks)
    var iters = 50
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var ctx = IntelGPUContext()
        ctx.close()
    var t1 = perf_counter_ns()
    print("Init/destroy x", iters, ":", ns_to_ms(t1 - t0), "ms")

    # 2. Repeated alloc/free churn (small + large, interleaved)
    var ctx = IntelGPUContext()
    var sizes = [64, 1024, 65536, 1048576, 8388608]
    var churn = 200
    t0 = perf_counter_ns()
    for i in range(churn):
        var sz = sizes[i % len(sizes)]
        var d = ctx.allocate_device(UInt64(sz))
        var h = ctx.allocate_host(UInt64(sz))
        ctx.free_device(d)
        ctx.free_host(h)
    t1 = perf_counter_ns()
    print("Alloc/free churn x", churn, ":", ns_to_ms(t1 - t0), "ms")

    # 3. Large single allocations up to 512MB
    var big = [134217728, 268435456, 536870912]
    for sz in big:
        var d = ctx.allocate_device(UInt64(sz))
        print("  Allocated device", sz // 1048576, "MB OK")
        ctx.free_device(d)

    # 4. Kernel launch flood
    var N = 1024
    var bytes = UInt64(N * 4)
    var ha = ctx.allocate_host(bytes)
    var hb = ctx.allocate_host(bytes)
    var hc = ctx.allocate_host(bytes)
    var da = ctx.allocate_device(bytes)
    var db = ctx.allocate_device(bytes)
    var dc = ctx.allocate_device(bytes)
    var kernel = Kernel(ctx.library(), ctx.context(), ctx.device(), ctx.command_list(), "./vector_add.spv", "vector_add")
    kernel.set_arg_pointer(0, da)
    kernel.set_arg_pointer(1, db)
    kernel.set_arg_pointer(2, dc)
    kernel.set_arg_value(3, 4, UInt64(N))
    kernel.set_group_size(256, 1, 1)
    var g = ZeGroupCount(4, 1, 1)

    var launches = 5000
    t0 = perf_counter_ns()
    for _ in range(launches):
        kernel.launch(g)
    ctx.synchronize()
    t1 = perf_counter_ns()
    print("Kernel launch flood x", launches, ":", ns_to_ms(t1 - t0), "ms")

    ctx.free_device(da); ctx.free_device(db); ctx.free_device(dc)
    ctx.free_host(ha); ctx.free_host(hb); ctx.free_host(hc)
    ctx.close()
    print("=== Stress test complete ===")

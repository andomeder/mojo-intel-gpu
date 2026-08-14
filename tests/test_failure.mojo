"""Failure-mode tests for mojo_intel_gpu - exercises error paths and edge cases."""
from mojo_intel_gpu import IntelGPUContext, Kernel
from mojo_intel_gpu.utils.detect import detect_intel_gpu

def main() raises:
    print("=== Failure Mode Test ===")
    var pass_count = 0
    var fail_count = 0

    if detect_intel_gpu():
        print("  [OK] detect_intel_gpu() returns True"); pass_count += 1
    else:
        print("  [FAIL] detect_intel_gpu() returned False"); fail_count += 1

    # Double close is idempotent
    var ctx = IntelGPUContext()
    ctx.close()
    try:
        ctx.close()
        print("  [OK] double close is safe"); pass_count += 1
    except:
        print("  [FAIL] double close raised"); fail_count += 1

    # Zero-size allocation should not crash
    var ctx2 = IntelGPUContext()
    try:
        var z = ctx2.allocate_device(UInt64(0))
        ctx2.free_device(z)
        print("  [OK] zero-size device alloc returned"); pass_count += 1
    except:
        print("  [INFO] zero-size device alloc raised (L0 error, not a crash)");
        pass_count += 1

    # Nonexistent SPIR-V file
    try:
        var k = Kernel(ctx2.library(), ctx2.context(), ctx2.device(), ctx2.command_list(), "./nonexistent.spv", "foo")
        print("  [FAIL] nonexistent SPIR-V did not raise"); fail_count += 1
    except:
        print("  [OK] nonexistent SPIR-V raised");
        pass_count += 1

    # Invalid SPIR-V (non-SPIRV file) must raise cleanly, not abort
    try:
        var k2 = Kernel(ctx2.library(), ctx2.context(), ctx2.device(), ctx2.command_list(), "./README.md", "foo")
        print("  [FAIL] invalid SPIR-V did not raise"); fail_count += 1
    except:
        print("  [OK] invalid SPIR-V raised cleanly");
        pass_count += 1

    # free_device already guards against null; a second free is UB, not tested
    var p = ctx2.allocate_device(UInt64(1024))
    ctx2.free_device(p)
    print("  [OK] free_device released allocation"); pass_count += 1

    ctx2.close()
    print("=== Failure test done: pass=", pass_count, " fail=", fail_count, " ===")
    if fail_count > 0:
        raise Error("Failure-mode test found issues")

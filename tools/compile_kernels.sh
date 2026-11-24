#!/bin/bash
# compile_kernels.sh — Compile OpenCL C kernels to SPIR-V
#
# Usage:
#   ./tools/compile_kernels.sh
#
# Requirements:
#   - clang (with SPIR-V target support)
#   - llvm-spirv
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR"

echo "=== Compiling OpenCL C Kernels to SPIR-V ==="
echo "Source directory: $SRC_DIR"

# Locate the toolchain. Prefer whatever is on PATH; otherwise fall back to an
# Intel oneAPI install, whose compiler/<ver>/bin/compiler/ directory ships a
# matched clang + llvm-spirv pair (no extra packages needed).
CLANG="$(command -v clang || true)"
LLVM_SPIRV="$(command -v llvm-spirv || true)"

if [ -z "$CLANG" ] || [ -z "$LLVM_SPIRV" ]; then
    for root in "${ONEAPI_ROOT:-/opt/intel/oneapi}" /opt/intel/oneapi; do
        [ -d "$root/compiler" ] || continue
        for cbin in "$root"/compiler/*/bin/compiler; do
            [ -d "$cbin" ] || continue
            if [ -z "$CLANG" ] && [ -x "$cbin/clang" ]; then
                CLANG="$cbin/clang"
            fi
            if [ -z "$LLVM_SPIRV" ] && [ -x "$cbin/llvm-spirv" ]; then
                LLVM_SPIRV="$cbin/llvm-spirv"
            fi
        done
        [ -n "$CLANG" ] && [ -n "$LLVM_SPIRV" ] && break
    done
fi

if [ -z "$CLANG" ] || [ -z "$LLVM_SPIRV" ]; then
    echo "Error: need 'clang' (with SPIR-V target) and 'llvm-spirv'" >&2
    echo "  Put them on PATH, or install Intel oneAPI, which provides a" >&2
    echo "  matched pair under compiler/<ver>/bin/compiler/." >&2
    exit 1
fi

echo "Using clang:      $CLANG"
echo "Using llvm-spirv: $LLVM_SPIRV"

# Compile each .cl file
COMPILED=0
FAILED=0

for cl_file in "$SRC_DIR"/*.cl; do
    if [ ! -f "$cl_file" ]; then
        continue
    fi

    base_name=$(basename "$cl_file" .cl)
    spv_file="$SRC_DIR/${base_name}.spv"
    bc_file="$SRC_DIR/${base_name}.bc"

    echo "Compiling: $base_name.cl"

    # Step 1: OpenCL C → LLVM bitcode
    #
    # `-c` is required. Without it the clang driver performs the SPIR-V *link*
    # step, which shells out to `spirv-link` — a binary shipped neither by
    # Intel oneAPI nor by current SPIRV-Tools/Fedora. The failure surfaces as a
    # misleading "posix_spawn failed" / "spirv-link command failed" error.
    #
    # The fully-qualified triple `spirv64-unknown-unknown` is used rather than
    # the bare `spirv64`: LLVM 21+ requires the explicit triple form. Both work
    # on clang 22 (verified), so the qualified form covers the wider range.
    if "$CLANG" -c -cl-std=CL3.0 -target spirv64-unknown-unknown -emit-llvm "$cl_file" -o "$bc_file" 2>/dev/null; then
        # Step 2: LLVM bitcode → SPIR-V
        if "$LLVM_SPIRV" "$bc_file" -o "$spv_file" 2>/dev/null; then
            echo "  ✓ Generated: ${base_name}.spv ($(stat -f%z "$spv_file" 2>/dev/null || stat -c%s "$spv_file") bytes)"
            rm -f "$bc_file"
            COMPILED=$((COMPILED + 1))
        else
            echo "  ✗ Failed: llvm-spirv conversion"
            rm -f "$bc_file"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ✗ Failed: clang compilation"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Compilation Summary ==="
echo "  Compiled: $COMPILED"
echo "  Failed:   $FAILED"

if [ $FAILED -gt 0 ]; then
    echo "  ✗ Some kernels failed to compile"
    exit 1
else
    echo "  ✓ All kernels compiled successfully"
fi

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

# Check for required tools
if ! command -v clang &> /dev/null; then
    echo "Error: clang not found"
    echo "Install with: sudo apt install clang"
    exit 1
fi

if ! command -v llvm-spirv &> /dev/null; then
    echo "Error: llvm-spirv not found"
    echo "Install with: sudo apt install llvm-spirv"
    exit 1
fi

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
    if clang -cl-std=CL3.0 -target spirv64 -emit-llvm "$cl_file" -o "$bc_file" 2>/dev/null; then
        # Step 2: LLVM bitcode → SPIR-V
        if llvm-spirv "$bc_file" -o "$spv_file" 2>/dev/null; then
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

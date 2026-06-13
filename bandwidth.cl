// bandwidth.cl — Memory bandwidth test kernels
//
// vec_copy: simple global-to-global copy. Used to measure kernel-side
// memory throughput (vs L0's native memcpy which uses the copy engine).

__kernel void vec_copy(
    __global const float* src,
    __global float* dst,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) {
        dst[i] = src[i];
    }
}

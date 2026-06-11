__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

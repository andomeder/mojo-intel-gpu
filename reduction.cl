__kernel void reduce_sum(__global const float *input,
                         __global float *output,
                         const int n) {
    __local float sdata[256];
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    int group_id = get_group_id(0);
    int block_size = get_local_size(0);

    // Load into shared memory
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write partial sum for this workgroup
    if (tid == 0) {
        output[group_id] = sdata[0];
    }
}

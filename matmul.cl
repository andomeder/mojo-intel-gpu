__kernel void matmul(__global const float *A,
                     __global const float *B,
                     __global float *C,
                     const int M,
                     const int N,
                     const int K) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N) {
        B[idx] = A[idx];
    }

}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    float *d_A, *d_B;
    int total = N * N;
    float size = N * N * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

} 

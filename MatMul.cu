#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
       int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float value = 0.0f;

        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * K + col];
        }

        C[row * K + col] = value;
}
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    float *d_A, *d_B, *d_C;

size_t sizeA = M * N * sizeof(float);
size_t sizeB = N * K * sizeof(float);
size_t sizeC = M * K * sizeof(float);


    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

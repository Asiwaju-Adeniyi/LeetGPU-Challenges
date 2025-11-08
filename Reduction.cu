#include <cuda_runtime.h>
#include <iostream>

__global__ void ReduceSum(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    if (i + blockDim.x < N) {
        sdata[tid] += input[i + blockDim.x];
    }
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Global accumulation
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {  

    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    size_t outputSize = sizeof(float);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, outputSize);
    cudaMemset(d_output, 0, outputSize);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    ReduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

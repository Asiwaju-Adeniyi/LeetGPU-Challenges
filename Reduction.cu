#include <cuda_runtime.h>
#include <iostream>

__global__ void blockReduceSum(const float *input, float *partial, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float x = 0.0f;
    if (i < N) x = input[i];
    if (i + blockDim.x < N) x += input[i + blockDim.x];
    sdata[tid] = x;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void finalReduceSum(const float *partial, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = tid;

    // Load partial results into shared memory
    sdata[tid] = (i < N) ? partial[i] : 0.0f;
    __syncthreads();

    // Reduce within a single block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) *output = sdata[0];
}

extern "C" void solve(const float* input, float* output, int N) {  
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    float *d_input, *d_partial, *d_output;
    size_t size = N * sizeof(float);
    size_t partialSize = blocksPerGrid * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_partial, partialSize);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // First pass: reduce per block
    blockReduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_partial, N
    );

    // Second pass: reduce all partial sums into one
    int threadsFinal = 256;
    int blocksFinal = 1;
    finalReduceSum<<<blocksFinal, threadsFinal, threadsFinal * sizeof(float)>>>(
        d_partial, d_output, blocksPerGrid
    );

    // Copy result back
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_output);
}

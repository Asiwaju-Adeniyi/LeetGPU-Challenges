#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > N/2) return;

    int opposite = N - 1 - idx;
    float temp = input[idx];
    input[idx] = input[opposite];
    input[opposite] = temp;
}

// input is device pointer
extern "C" void solve(float* input, int N) {

    float *d_input;
    float size = N * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_input, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(input, d_input, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
}

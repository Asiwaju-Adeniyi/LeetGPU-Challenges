#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {

    float *d_input, *d_output;
    float size = N * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

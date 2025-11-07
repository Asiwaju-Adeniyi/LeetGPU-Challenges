#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
        float x1 = input[idx];
        float x2 = input[idx + halfN];
        float sigmoid_x1 = 1.0f / (1.0f + expf(-x1));
        float silu_x1 = x1 * sigmoid_x1;
        output[idx] = silu_x1 * x2;
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    float *d_input, *d_output;
    size_t input_size = N * sizeof(float);
    size_t output_size = halfN * sizeof(float);


    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, halfN);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

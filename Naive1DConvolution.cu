#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (i >= output_size) return;

    float sum = 0.0f;

    for (int j = 0; j < kernel_size; ++j) {
        sum += input[i + j] * kernel[j];
    }

    output[i] = sum;

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    float *d_input, *d_kernel;
    float *d_output;

    size_t inputSize = input_size * sizeof(float);
    size_t kernelSize = kernel_size * sizeof(float);
    size_t outputSize = (input_size - kernel_size + 1) * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, outputSize);

    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, input_size, kernel_size);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
    C[idx] = A[idx] + B[idx];
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent, 0);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Computation performed in " << gpuDuration << "ms." << std::endl;

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    
    std::vector<float> h_A = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> h_B = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> h_C(4);


    solve(h_A.data(), h_B.data(), h_C.data(), 4);

    for (float val : h_C) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;

}

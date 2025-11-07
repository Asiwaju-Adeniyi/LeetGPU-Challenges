#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x; 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; 

    int count = 0;
    while (i < N) {
        if (input[i] == K) {
            count++;
        }

        i += stride;
    } 

    sdata[tid] = count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {

    int *d_input, *d_output;
    int size = N * sizeof(int);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(int));
    cudaMemset(d_output, 0, sizeof(int));

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_input, d_output, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

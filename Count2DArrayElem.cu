#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    extern __shared__ int shared_count[];    // dynamic sized per-block

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    int value = 0;
    if (row < N && col < M) {
        int idx = row * M + col;
        if (input[idx] == K) {
            value = 1;
        }
    }

    // store each thread's 0/1 into shared memory
    shared_count[idx_in_block] = value;
    __syncthreads();

    // block-level reduction (requires sync each iteration)
    for (int stride = numThreads / 2; stride > 0; stride >>= 1) {
        if (idx_in_block < stride) {
            shared_count[idx_in_block] += shared_count[idx_in_block + stride];
        }
        __syncthreads();
    }

    // thread 0 adds the block's total to the global output
    if (idx_in_block == 0) {
        atomicAdd(output, shared_count[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    int *d_input = nullptr, *d_output = nullptr;
    size_t size = (size_t)M * (size_t)N * sizeof(int);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(int));
    cudaMemset(d_output, 0, sizeof(int));

    // copy input
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // compute shared memory size: one int per thread in a block
    size_t sharedMemSize = threadsPerBlock.x * threadsPerBlock.y * sizeof(int);

    // launch
    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N, M, K);
    cudaDeviceSynchronize();

    // copy result back (single int)
    cudaMemcpy(output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

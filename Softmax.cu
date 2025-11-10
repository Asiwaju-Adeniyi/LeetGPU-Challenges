#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits> 

__global__ void reduce_max(const float* input, float* max_val, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (i < N) ? input[i] : -CUDART_INF_F;
    __syncthreads();

    // Parallel reduction (max)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        max_val[blockIdx.x] = sdata[0];
}

__global__ void reduce_sum(const float* input, float* sum_val, float max_val, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < N)
        val = expf(input[i] - max_val);

    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction (sum)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        sum_val[blockIdx.x] = sdata[0];
}

__global__ void softmax_kernel(const float* input, float* output, float max_val, float sum_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = expf(input[idx] - max_val) / sum_val;
}

extern "C" void solve(const float* input, float* output, int N) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const size_t smem = threads * sizeof(float);

    float *d_input, *d_output;
    float *d_partial, *d_sum;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMalloc(&d_sum, blocks * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // ---- Stage 1: find max ----
    reduce_max<<<blocks, threads, smem>>>(d_input, d_partial, N);
    cudaDeviceSynchronize();

    float *h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float max_val = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < blocks; ++i)
        if (h_partial[i] > max_val) max_val = h_partial[i];

    // ---- Stage 2: compute sum(exp(x - max)) ----
    reduce_sum<<<blocks, threads, smem>>>(d_input, d_sum, max_val, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float sum_val = 0.0f;
    for (int i = 0; i < blocks; ++i)
        sum_val += h_partial[i];

    // ---- Stage 3: normalize ----
    softmax_kernel<<<blocks, threads>>>(d_input, d_output, max_val, sum_val, N);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free
    delete[] h_partial;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    cudaFree(d_sum);
}

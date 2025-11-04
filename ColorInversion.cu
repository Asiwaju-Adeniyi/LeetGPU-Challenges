#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= width * height) return;

    unsigned char* pixel = image + (idx * 4);

    pixel[0] = 255 - pixel[0];
    pixel[1] = 255 - pixel[1];
    pixel[2] = 255 - pixel[2];

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {

    unsigned char *image_input;
    

    int size = width * height * 4 * sizeof(char);

    cudaMalloc(&image_input, size);
    

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(image_input, image, size, cudaMemcpyHostToDevice);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image_input, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(image, image_input, size, cudaMemcpyDeviceToHost);

    cudaFree(image_input);
}

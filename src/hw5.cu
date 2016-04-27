/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cu
 * April 26, 2016
 */

#include "../include/hw5.cuh"

__global__ void median_filter(float *input, float* output, const uint32_t width, const uint32_t height, const uint32_t filter_size) {
    // Get x and y location
    uint32_t x           = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y           = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t window_size = filter_size * filter_size;
    uint32_t pivot =  window_size / 2;
    uint32_t range = filter_size / 2;

    // error check
    if((x >= width - range) || (y >= height - range) || (x - range) <= 0 || (y - range) <= 0) {
        return;
    }
    
    // Allocate neighborhood
    float *neighborhood = (float *)malloc(window_size * sizeof(float));

    int i = 0;
    // Fill neighborhood accordinly
    for(int current_x = x - range; current_x <= x + range; current_x++) {
        for(int current_y = y - range; current_y <= y + range; current_y++) {
            neighborhood[i] = input[(current_y * width) + current_x];
            ++i;
        }
    }

    // Bubble sort for first half
    for(int k = 0; k <= pivot; k++) {
        int min_index = k;

        for(int j = k + 1; j < window_size; j++) {
            if(neighborhood[j] < neighborhood[min_index]) {
                min_index = j;
            }
        }

        // Swap
        float temp = neighborhood[k];
        neighborhood[k] = neighborhood[min_index];
        neighborhood[min_index] = temp;
    }

    output[(y * width) + x] = neighborhood[pivot];

    free(neighborhood);
}// end median_filter


bool hw5_cuda::device_load(float **host_image, uint32_t width, uint32_t height, uint32_t filter_size, float** output){
    unsigned int size = width * height * sizeof(float);

    // Alloc device 
    float* device_image = NULL;
    float* device_output_image = NULL;

    *output = (float *)malloc(size);

    cudaMalloc((void**)&device_image, size);
    cudaMalloc((void**)&device_output_image, size);
    cudaMemcpy(device_image, *host_image, size, cudaMemcpyHostToDevice);
    
    // Attempt to fix border around image
    cudaMemcpy(device_output_image, *host_image, size, cudaMemcpyHostToDevice);

    // dim defined in hw5.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // run filter
    median_filter<<<dimGrid, dimBlock, 0>>>(device_image, device_output_image, width, height, filter_size);

    cudaMemcpy(*output, device_output_image, size, cudaMemcpyDeviceToHost);

    return true;

}// end hw5_cuda::device_load
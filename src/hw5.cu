/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cu
 * April 26, 2016
 */

#include "../include/hw5.cuh"

__global__ void median_filter( uint8_t *input, uint32_t *width, uint32_t *height, uint32_t *filter_size, uint8_t *outputData ){
    // (ppms are 1-byte per pixel)

    // calculate normalized texture coordinates
    uint32_t x           = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y           = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t thread_id   = threadIdx.y * blockDim.y + threadIdx.x;

    // for now use a double array to represent all windows for all threads
    uint32_t window_size = (*filter_size) * (*filter_size);

    // for now we'll build for a 3x3 window
    // don't need int for array, we really want uint8_t but the compiler won't let us?
    __shared__ int window[DIM_X*DIM_Y][9];



    // error check
    if( x > *width || y > *height ){
        return;
    }

    // populate window

    #if 1   // 1 for fill window dynamic-sized, 0 for fill only 3x3 windows (for testing)
    for( uint32_t i = 0; i < window_size; i++){
        uint32_t local_x = i / *width;
        uint32_t local_y = i % *height;

        // big dumb slow decision tree to find edge cases that breaks things
        if( y == 0 ){
            if( x == 0 ){ // top-left corner case
                window[thread_id][i] = 0.0;
            }
            else if( x == *width-1 ){ // bottom left case
                window[thread_id][i] = 0.0;
            }
            else{ // left edge case
                window[thread_id][i] = 0.0;
            }
        }
        else if( x == 0 ){ // top edge case
            window[thread_id][i] = 0.0;
        }
        else if( x == *width-1 ){ // bottom edge case
            window[thread_id][i] = 0.0;
        }
        else if( y == *height-1 ){
            if( x == 0 ){ // top right case
                window[thread_id][i] = 0.0;
            }
            else if( x == *width-1 ){ // bottom right case
                window[thread_id][i] = 0.0;
            }
            else{ // right edge case
                window[thread_id][i] = 0.0;
            }
        }
        // non edge cases down here
        else{
            window[thread_id][i] = (uint8_t)input[ (y + (local_x - x) ) * (*width) + (x + (local_y - y))];
        }

    }// end for
    #else
    window[tid][0] = (y == 0 || x==0)               ? 0.0f : input[(y-1) * (*width) + x-1];
    window[tid][1] = (y == 0)                       ? 0.0f : input[(y-1) * (*width) + x];
    window[tid][2] = (y == 0 || x == (*width) - 1)  ? 0.0f : input[(y-1) * (*width) + x+1];
    window[tid][3] = (x == 0)                       ? 0.0f : input[y * (*width) + x-1];
    window[tid][4] = input[y*(*width)+x];
    window[tid][5] = (x == (*width) - 1)            ? 0.0f : input[y * (*width) + x+1];
    window[tid][6] = (y == (*height) - 1||x==0)     ? 0.0f : input[(y+1) * (*width) + x-1];
    window[tid][7] = (y == (*height) - 1)           ? 0.0f : input[(y+1) * (*width) + x];
    window[tid][8] = (y == (*height) - 1||x== (*width) -1) ? 0.0f:input[(y+1) * (*width) + x+1];
    #endif

    syncthreads();

    #if 0   // 1 for dynamic-sized window, 0 for fill only 3x3 windows (for testing)

    #else
    // Order elements (only half of them)
    for (unsigned int j=0; j<(window_size/2)+1; ++j){
        // Find position of minimum element
        int min = j;
        for (unsigned int l=j+1; l<9; ++l) {
            if (window[thread_id][l] < window[thread_id][min]) {
                min = l;
            }
        }

        // Put found minimum element in its place
        const uint8_t temp=window[thread_id][j];
        window[thread_id][j]=window[thread_id][min];
        window[thread_id][min]=temp;

        syncthreads();
    }

    input[y * (*width) + x] = window[thread_id][4];
    #endif

}// end median_filter


bool hw5_cuda::device_load( uint8_t **image, uint32_t *width, uint32_t *height, uint32_t *filter_size ){
    unsigned int size = (*width) * (*height) * sizeof(uint8_t);

    // Allocate array and copy image data
    cudaArray *cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    //checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, *width, *height));
    //checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, image, size, cudaMemcpyHostToDevice));
    cudaMallocArray(&cuArray, &channelDesc, *width, *height);
    cudaMemcpyToArray(cuArray, 0, 0, *image, size, cudaMemcpyHostToDevice);

    // dim defined in hw5.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(*width / dimBlock.x, *height / dimBlock.y, 1);

    // Allocate device memory for result
    uint8_t *return_data = NULL;
    //checkCudaErrors(cudaMalloc((void **) &return_data, size));
    cudaMalloc((void **) &return_data, size);   // compiler complaining about checkCudaErrors

    // run filter
    median_filter<<<dimGrid, dimBlock, 0>>>( *image, width, height, filter_size, return_data );


    return true;
}// end hw5_cuda::device_load
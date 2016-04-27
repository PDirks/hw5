/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cu
 * April 26, 2016
 */

#include "../include/hw5.cuh"

__global__ void median_filter( uint8_t *image, uint32_t *width, uint32_t *height, uint32_t *filter_size ){

    // calculate normalized texture coordinates
    uin32_t x           = blockIdx.x * blockDim.x + threadIdx.x;
    uin32_t y           = blockIdx.y * blockDim.y + threadIdx.y;
    uin32_t thread_id   = threadIdx.y * blockDim.y + threadIdx.x;

    // for now use a double array to represent all windows for all threads
    const uint32_t window_size = (*filter_size) * (*filter_size);
    __shared__ char window[DIM_X*DIM_Y][];    // (ppms are 1-byte per pixel)

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
                window[thread_id][i] = 0f;
            }
            else if( x == width-1 ){ // bottom left case
                window[thread_id][i] = 0f;
            }
            else{ // left edge case
                window[thread_id][i] = 0f;
            }
        }
        else if( x == 0 ){ // top edge case
            window[thread_id][i] = 0f;
        }
        else if( x == width-1 ){ // bottom edge case
            window[thread_id][i] = 0f;
        }
        else if( y == height-1 ){
            if( x == 0 ){ // top right case
                window[thread_id][i] = 0f;
            }
            else if( x == width-1 ){ // bottom right case
                window[thread_id][i] = 0f;
            }
            else{ // right edge case
                window[thread_id][i] = 0f;
            }
        }
        // non edge cases down here
        else{
            window[thread_id][i] = input[ (y + (local_x - x) ) * width + (x + (local_y - y))];
        }

    }// end for
    #elseif
    window[tid][0] = (y==0||x==0)               ? 0.0f : input[(y-1)*DATA_W+x-1];
    window[tid][1] = (y==0)                     ? 0.0f : input[(y-1)*DATA_W+x];
    window[tid][2] = (y==0||x==DATA_W-1)        ? 0.0f : input[(y-1)*DATA_W+x+1];
    window[tid][3] = (x==0)                     ? 0.0f : input[y*DATA_W+x-1];
    window[tid][4] = input[y*DATA_W+x];
    window[tid][5] = (x==DATA_W-1)              ? 0.0f : input[y*DATA_W+x+1];
    window[tid][6] = (y==DATA_H-1||x==0)        ? 0.0f : input[(y+1)*DATA_W+x-1];
    window[tid][7] = (y==DATA_H-1)              ? 0.0f : input[(y+1)*DATA_W+x];
    window[tid][8] = (y==DATA_H-1||x==DATA_W-1) ? 0.0f:input[(y+1)*DATA_W+x+1];
    #endif

    syncthreads();

    #if 0   // 1 for dynamic-sized window, 0 for fill only 3x3 windows (for testing)

    #elseif
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

    input[y * width + x] = window[thread_id][4];
    #endif

}// end median_filter


bool hw5_cuda::device_load( uint8_t *image, uint32_t *width, uint32_t *height, uint32_t *filter_size ){

    // dim defined in hw5.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(*width / dimBlock.x, *height / dimBlock.y, 1);

    // run filter
    median_filter<<<dimGrid, dimBlock, 0>>>( image, width, height, filter_size );



}// end hw5_cuda::device_load
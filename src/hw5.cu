/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cu
 * April 26, 2016
 */

#include "../include/hw5.cuh"

__global__ void median_filter(uint8_t *input, const uint32_t width, const uint32_t height, const uint32_t filter_size) {
    // Get x and y location
    uint32_t x           = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y           = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t window_size = filter_size * filter_size;
    uint32_t pivot =  window_size / 2;
    int range = filter_size / 2;

    // error check
    if((x >= width - range) || (y >= height - range) || ((int)x - range) < 0 || ((int)y - range) < 0) {
        return;
    }
    
    // Allocate neighborhood
    uint8_t* neighborhood = new uint8_t[window_size];

    int i = 0; 
    
    // Fill neighborhood accordinly
    for(int current_x = x - range; current_x <= x + range; current_x++) {
        for(int current_y = y - range; current_y <= y + range; current_y++) {
            neighborhood[i] = input[(current_y * width) + current_x];
            ++i;
        }
    }

    // Bubble sort for first half, choosen over insertion sort due to stop once median found
    for(int k = 1; k <= pivot; k++) {
        int min_index = k;

        for(int j = k + 1; j < window_size; j++) {
            if(neighborhood[j] < neighborhood[min_index]) {
                min_index = j;
            }
        }

        // Swap
        uint8_t temp = neighborhood[k];
        neighborhood[k] = neighborhood[min_index];
        neighborhood[min_index] = temp;
    }

    // Update the image
    input[(y * width) + x] = neighborhood[pivot];

    free(neighborhood);
    
}// end median_filter

double hw5_cuda::device_load(uint8_t **host_image, uint32_t width, uint32_t height, uint32_t filter_size, uint8_t** output){

    unsigned int size = width * height * sizeof(uint8_t);

    // Alloc device 
    uint8_t* device_image = NULL;

    *output = (uint8_t *)malloc(size);

    cudaMalloc((void**)&device_image, size);
    cudaMemcpy(device_image, *host_image, size, cudaMemcpyHostToDevice);

    // dim defined in hw5.cuh, we'll start with a 8x8x1
    dim3 dimBlock(DIM_X, DIM_Y, DIM_Z);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // run filter
    median_filter<<<dimGrid, dimBlock, 0>>>(device_image, width, height, filter_size);

    cudaMemcpy(*output, device_image, size, cudaMemcpyDeviceToHost);



    return sdkGetTimerValue(&timer);

}// end hw5_cuda::device_load

void hw5_cuda::cpu_filter( uint8_t *host_image, uint8_t *cpu_image, uint32_t width, uint32_t height, uint32_t filter_size ){
    uint32_t window_size = filter_size * filter_size;
    int range = (filter_size)/ 2;

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            uint8_t neighborhood[window_size];
            uint32_t neighborhood_index = 0;
            // zero out neighborhood array
            for (uint32_t k = 0; k < window_size; k++){
                neighborhood[k] = 0;
            }

            // populate filter array
            //std::cout << "populate" << std::endl;
            for(int current_x = x - range; current_x <= x + range; current_x++) {
                for (int current_y = y - range; current_y <= y + range; current_y++) {
                    if((x >= width - range) || (y >= height - range) || ((int)x - range) < 0 || ((int)y - range) < 0) {
                        continue;
                    }
                    neighborhood[neighborhood_index] = host_image[(current_y * (int)width) + current_x];
                    neighborhood_index++;
                    //std::cout << "boop" << std::endl;
                }// end inner populate
            }// end outer populate

            //std::cout << "sorting" << std::endl;
            std::sort( neighborhood, neighborhood + window_size );
            //std::cout << "sorting done" << std::endl;
            cpu_image[x + width * y ] = neighborhood[range];
        } // end col for
    }// end row for

}// end cpu_filter

double hw5_cuda::image_filter_error( uint8_t **host_image, uint8_t *gpu_image, uint32_t width, uint32_t height, uint32_t filter_size ){
    char cpu_file[] = "out_cpu.pgm";
    uint32_t window_size = height * width;

    /*
     * create cpu_image
     */
    uint8_t *cpu_image = (uint8_t*)calloc( window_size, sizeof( uint8_t ) );
    if( cpu_image == NULL ){
        return -1;
    }
    cpu_filter( *host_image, cpu_image, width, height, filter_size );
    /*
     * compare pixels for errors
     */
    uint32_t error_count = 0;
    for( uint32_t i = 0; i < window_size; i++ ){
        if( cpu_image[i] != gpu_image[i] ){
            error_count++;
        }
    }
    /*
     * save cpu image file
     */
    if(sdkSavePGM( cpu_file, cpu_image, width, height ) == false){
        return -1;
    }

    //free(cpu_image);

    return (double) (error_count / window_size);
}// end image_filter_error

void hw5_cuda::timerStart(){
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
}
void hw5_cuda::timerStop(){
    sdkStopTimer(&timer);
    sdkDeleteTimer(&timer);
}
double hw5_cuda::getTime(){
    return sdkGetTimerValue(&timer);
}
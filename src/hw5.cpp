/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cpp
 * April 26, 2016
 */


#include "../include/util.hpp"
#include "../include/hw5.cuh"
#include "../include/sdkHelper.h"

/*
 * input:
 *      ./hw5 [int filter size] [input file name] [output file name]
 * example:
 *      ./hw5 3 ../data/lena.ppm ../out.ppm
 */
int main(int argc, char *argv[]){

    /*
     * Check input parameters
     */
    if( argc != 4){
        err( "INSUFFICIENT PARAMETERS, format: [int filter size] [input file name] [output file name]" );
    }

    uint32_t filter_size      = atoi(argv[1]);
    std::string input_file    = argv[2];
    std::string output_file   = argv[3];

    if( filter_size <= 0 ){
        err( "FILTER SIZE NEEDS TO BE GREATER THAN 0" );
    }

    /*
     * Load input file
     */
    uint32_t width    = 0;
    uint32_t height   = 0;
    float *host_image = NULL;

    // Load Image on host
    if(sdkLoadPGM( input_file.c_str(), &host_image, &width, &height ) == false){
        err( "ERROR ON PPM LOAD" );
    }
    float *output = NULL;

    /*
     * Push file data to device memory
     */
    if(hw5_cuda::device_load( &host_image, width, height, filter_size, &output ) == false){
        err( "ERROR ON LOAD TO DEVICE" );
    }

    if(sdkSavePGM( output_file.c_str(), output, width, height ) == false){
        err( "ERROR ON PPM LOAD" );
    }
    /*
     * Cleanup
     */
    free(host_image);
    free(output);

    return 0;
}// end main
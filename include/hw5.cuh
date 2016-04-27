/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cuh
 * April 26, 2016
 */

namespace hw5_cuda{

    #include <stdio.h>
    #define _TEST_KERNEL_CU_

    // constants
    #define DIM_X           8
    #define DIM_Y           8
    #define DIM_Z           1

    bool hw5_cuda::device_load( uint8_t *image, uint32_t *width, uint32_t *height, uint32_t *filter_size );

}
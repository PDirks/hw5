/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * hw5.cuh
 * April 26, 2016
 */
#include "../include/sdkHelper.h"
#include <stdint.h> // uint
#include <stdio.h>
#include <cuda_runtime.h>

#define _TEST_KERNEL_CU_

// constants
#define DIM_X           8
#define DIM_Y           8
#define DIM_Z           1

class hw5_cuda {
public:
    StopWatchInterface *timer;

    double device_load(uint8_t **image, uint32_t width, uint32_t height, uint32_t filter_size, uint8_t **output);
    void cpu_filter( uint8_t *host_image, uint8_t **gpu_image, uint32_t width, uint32_t height, uint32_t filter_size );
    double image_filter_error( uint8_t **host_image, uint8_t **gpu_image, uint32_t width, uint32_t height, uint32_t filter_size );
    void timerStart();
    void timerStop();
    double getTime();
};

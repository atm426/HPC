

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "utils.h"
#include <algorithm>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <sm_30_intrinsics.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 1024
void Check_CUDA_Error(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}
__global__ void inner_prod(const double* x, const double* y, double* sum, long N)
{
    __shared__ double smem_x[BLOCK_SIZE];
    __shared__ double smem_y[BLOCK_SIZE];
    __shared__ double smem_sum[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) {
        smem_x[threadIdx.x] = x[idx];
        smem_y[threadIdx.x] = y[idx];
    }
    else {
        smem_x[threadIdx.x] = 0;
        smem_y[threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x < 512) smem_sum[threadIdx.x] += smem_y[threadIdx.x + 512] * smem_x[threadIdx.x + 512];
    __syncthreads();
    if (threadIdx.x < 256) smem_sum[threadIdx.x] += smem_y[threadIdx.x + 256] * smem_x[threadIdx.x + 256];
    __syncthreads();
    if (threadIdx.x < 128) smem_sum[threadIdx.x] += smem_y[threadIdx.x + 128] * smem_x[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x < 64) smem_sum[threadIdx.x] += smem_y[threadIdx.x + 64] * smem_x[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32) {
        smem_sum[threadIdx.x] += smem_y[threadIdx.x + 32] * smem_x[threadIdx.x + 32];
        __syncwarp();
        smem_sum[threadIdx.x] += smem_y[threadIdx.x + 16] * smem_x[threadIdx.x + 16];
        __syncwarp();
        smem_sum[threadIdx.x] += smem_y[threadIdx.x + 8] * smem_x[threadIdx.x + 8];
        __syncwarp();
        smem_sum[threadIdx.x] += smem_y[threadIdx.x + 4] * smem_x[threadIdx.x + 4];
        __syncwarp();
        smem_sum[threadIdx.x] += smem_y[threadIdx.x + 2] * smem_x[threadIdx.x + 2];
        __syncwarp();
        if (threadIdx.x == 0) smem_sum[blockIdx.x] = smem_sum[0]+smem_sum[1];
   

        //printf("hello from block %d and thread %d / %d \n", blockIdx.x, threadIdx.x, blockDim.x);
        //printf("inner product from GPU = %f \n", sum[blockIdx.x]);
    }
    *sum += smem_sum[blockIdx.x];
}
__global__ void reduction_kernel2(double* sum, const double* a, long N) {
    __shared__ double smem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < N) smem[threadIdx.x] = a[idx];
    else smem[threadIdx.x] = 0;

    __syncthreads();
    if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
    __syncthreads();
    if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
    __syncthreads();
    if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
    __syncthreads();
    if (threadIdx.x < 64) smem[threadIdx.x] += smem[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32) {
        smem[threadIdx.x] += smem[threadIdx.x + 32];
        __syncwarp();
        smem[threadIdx.x] += smem[threadIdx.x + 16];
        __syncwarp();
        smem[threadIdx.x] += smem[threadIdx.x + 8];
        __syncwarp();
        smem[threadIdx.x] += smem[threadIdx.x + 4];
        __syncwarp();
        smem[threadIdx.x] += smem[threadIdx.x + 2];
        __syncwarp();
        if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
    }
}
double inner_prod_cpu(double *x, double *y, long N) {
    double sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

int main(int argc, char **argv)
{
    // filling arrays for inner product
    long const N = 1000000;
    double* x;
    double* y;
    cudaMallocHost((void**)&x, N * sizeof(double));
    cudaMallocHost((void**)&y, N * sizeof(double));
    
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
    }

    Timer tt;
    // CPU version, for reference
    tt.tic();
    double sum_ref = inner_prod_cpu(x, y, N);
    printf("CPU: t = %f seconds ; sum = %f \n", tt.toc(), sum_ref);

    // allocating memory for GPU
    double sum;
    double *x_d, *y_d, *sum_d;
    cudaMalloc(&x_d, N * sizeof(double));
    cudaMalloc(&y_d, N * sizeof(double));
    long N_work = 1;
    for (long i = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE); i > 1; i = (i + BLOCK_SIZE - 1) / (BLOCK_SIZE)) N_work += i;
    cudaMalloc(&sum_d, N_work * sizeof(double)); // extra memory buffer for reduction across thread-blocks

    // copying vectors to GPU
    cudaMemcpyAsync(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(y_d, y, N * sizeof(double), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    long Nb = (N + BLOCK_SIZE - 1) / (BLOCK_SIZE);
    // GPU version
    tt.tic();

    // max thread number per block is BS = 1024 
    // << # of blocks, # of threads per block >>
    // # blocks =( N + BS -1 )/ BS; # threads/block = BS
    // so that # threads ~ N (if N > BS)
    inner_prod<<<Nb, BLOCK_SIZE>>>(x_d, y_d, sum_d, N);
    
    cudaMemcpyAsync(&sum, sum_d, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("GPU: t = %f seconds ; sum = %f \n", tt.toc(), sum);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(sum_d);
    cudaFreeHost(x);
    cudaFreeHost(y);
    return 0;
}

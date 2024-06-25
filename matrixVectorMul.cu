#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <unistd.h>


// Kernel to perform matrix-vector multiplication
__global__ void matrixVectorMul(int* A, int* B, int* C, int n, int threadComputeCount) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int threadStartIndex = row * threadComputeCount;
    int threadEndIndex = threadStartIndex + threadComputeCount;

    if (threadStartIndex < n) {
        int sum = 0;
        for (int i = threadStartIndex; i < threadEndIndex && i < n; i++) {
            sum = 0;
            //long long start = clock64(); while(clock64()< (start+100000000));
            for (int j = 0; j < n; j++) {
                sum += A[i * n + j] * B[j];
            }
            C[i] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N;
    int blocks;
    int threads;
    struct timeval t1, t2;

    int *h_A, *h_B, *h_C, *h_CAfterGPU; // Host data
    int *d_A, *d_B, *d_C; // Device data

    if(argc != 4) {
        printf("wrong usage, required usage: matrixv num blocks threads\n");
            exit(1);
    }

    // assuming N is a multiple of number of threads of two configurations (multiple of 256)
    N = atoi(argv[1]);
    blocks = atoi(argv[2]);
    threads = atoi(argv[3]);

    // number of rows each thread needs to execute
    int threadCompute = N / (blocks*threads);

    // Allocate memory on the host
    h_A = (int*)malloc(N * N * sizeof(int));
    if(!h_A) {
        printf("Cannot allocate array h_A of %d elements\n", N * N);
            exit(1);
    }
    h_B = (int*)malloc(N * sizeof(int));
    if(!h_B) {
        printf("Cannot allocate array h_B of %d elements\n", N);
            exit(1);
    }
    h_C = (int*)malloc(N * sizeof(int));
    if(!h_C) {
        printf("Cannot allocate array h_C of %d elements\n", N);
            exit(1);
    }
    h_CAfterGPU = (int*)malloc(N * sizeof(int));
    if(!h_CAfterGPU) {
        printf("Cannot allocate array h_CAfterGPU of %d elements\n", N);
            exit(1);
    }

    // assigning random values to matrix A and vector B
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            h_A[i * N + j] = rand() % N;
        }
    }

    for(int i = 0; i < N; ++i) {
        h_B[i] = rand() % N;
    }

    // Using wall clock time to measure time using gettimeofday function
    // clock_t a = start() and end() give the CPU time - which meansures the time during which CPU is busy
    // This cannot be applied to measuring GPU execution time since CPU is idle during such times

    // measuring sequential execution time
    // gettimeofday(&t1, 0);

    // for(int i = 0; i < N; ++i) {
    //     int sum = 0;
    //     for(int j = 0; j < N; ++j) {
    //         sum += h_A[i * N + j] * h_B[j];
    //     }
    //     h_C[i] = sum;
    // }

    // gettimeofday(&t2, 0);
    // double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    // printf("Sequential execution time:  %3.3f ms \n", time);

    // Allocate memory on the device
    cudaMalloc(&d_A, N * N * sizeof(int));
    if(!d_A) {
        printf("Cannot allocate array d_A of %d elements\n", N * N);
            exit(1);
    }
    cudaMalloc(&d_B, N * sizeof(int));
    if(!d_B) {
        printf("Cannot allocate array d_B of %d elements\n", N);
            exit(1);
    }
    cudaMalloc(&d_C, N * sizeof(int));
    if(!d_C) {
        printf("Cannot allocate array d_C of %d elements\n", N);
            exit(1);
    }

    // Define the grid and block dimensions
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);

    // start measuring gpu execution time
    gettimeofday(&t1, 0);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    matrixVectorMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, threadCompute);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_CAfterGPU, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("%d:%.3f\n", N, time);

    // checking if final vector computed by gpu matches the one computed by cpu
    // for(int i = 0; i < N; ++i) {
    //     if(h_C[i] != h_CAfterGPU[i]) {
    //         printf("Mismatching elements - %d, %d\n", h_C[i], h_CAfterGPU[i]);
    //     }
    // }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_CAfterGPU);

    return 0;
}

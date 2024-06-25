#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <unistd.h>


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


__global__ void matSum(float* S, float* A, float* B, int N) {
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = i*N + j;

  if (tid < N*N) {
    S[tid] = A[tid] + B[tid];
  }
}


// Fills a vector with random float entries.
void randomInit(float* data, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int tid = i*N+j;
      data[tid] = (float)drand48();
    }
  }
}


int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Syntax: %s <matrix size> \n", argv[0]);
    return EXIT_FAILURE;
  }


  int N = atoi(argv[1]);

  // set seed for drand48()
  srand48(42);

  // allocate host memory for matrices A and B
  float* A = (float*) malloc(N * N * sizeof(float));
  float* B = (float*) malloc(N * N * sizeof(float));
  float* S = (float*) malloc(N * N * sizeof(float));

  // initialize host matrices
  randomInit(A, N);
  randomInit(B, N);

  // allocate device matrices (linearized)
  float* dev_A = NULL;
  float* dev_B = NULL;
  float* dev_S = NULL;
  checkCuda( cudaMalloc((void**) &dev_A, N * N * sizeof(float)) );
  checkCuda( cudaMalloc((void**) &dev_B, N * N * sizeof(float)) );
  checkCuda( cudaMalloc((void**) &dev_S, N * N * sizeof(float)) );

  // copy host memory to device
  checkCuda( cudaMemcpy(dev_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(dev_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice) );

  // execute the kernel
  int Tile_Width = 8;

  int GridSize = (N + Tile_Width-1) / Tile_Width;
  dim3 gridDim(GridSize, GridSize);
  dim3 blockDim(Tile_Width, Tile_Width);

  struct timeval t1, t2;
  gettimeofday(&t1, 0);

  matSum<<< gridDim, blockDim >>>(dev_S, dev_A, dev_B, N);
  cudaDeviceSynchronize();

  // copy result from device to host
  checkCuda( cudaMemcpy( S, dev_S, N * N * sizeof(float),cudaMemcpyDeviceToHost) );

  gettimeofday(&t2, 0);
  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("%d:%.3f\n", N, time);

  // clean up memory
  free(A);
  free(B);
  free(S);
  checkCuda( cudaFree(dev_A) );
  checkCuda( cudaFree(dev_B) );
  checkCuda( cudaFree(dev_S) );

  return 0;
}


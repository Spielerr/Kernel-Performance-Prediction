#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>


__global__ void addvector(int *, int *, int *, int);

int main(int argc, char *argv[])
{

    int i;
    int num = 0; // number of elements in the arrays
    int * a, *b, *c; // arrays at host
    int * ad, *bd, *cd; // arrays at device
    int THREADS = 0; // user decides number of threads per block

    if(argc != 3){
        printf("usage: addvec numelements num_threads\n");
        printf("cpu_or_gpu:  0 = CPU, 1  = GPU\n");
        exit(1);
    }

    num = atoi(argv[1]);
    THREADS = atoi(argv[2]);

    a = (int *)malloc(num*sizeof(int));
    if(!a){
        printf("Cannot allocate array a with %d elements\n", num);
        exit(1);
    }


    b = (int *)malloc(num*sizeof(int));
    if(!b){
        printf("Cannot allocate array b with %d elements\n", num);
        exit(1);
    }


    c = (int *)malloc(num*sizeof(int));
    if(!c){
        printf("Cannot allocate array c with %d elements\n", num);
        exit(1);
    }


    //Fill out arrays a and b with some random numbers
    srand(time(0));
    for( i = 0; i < num; i++)
    {
        a[i] = rand() % num;
        b[i] = rand() % num;
    }

    //Now zero C[] in preparation for GPU version
    for( i = 0; i < num; i++)
        c[i] = 0;


    int numblocks;
    int threadsperblock;

    if( (num % THREADS) == 0 )
        numblocks =num / THREADS ;
    else
        numblocks = (num/THREADS)>0? (num/THREADS)+1:1 ;
    threadsperblock = THREADS;

    // printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock);

    //assume a block can have THREADS threads
    dim3 grid(numblocks, 1, 1);
    dim3 block(threadsperblock, 1, 1);

    cudaMalloc((void **)&ad, num*sizeof(int));
    if(!ad)
    { printf("cannot allocated array ad of %d elements\n", num);
    exit(1);
    }


    cudaMalloc((void **)&bd, num*sizeof(int));
    if(!bd)
    {printf("cannot allocated array bd of %d elements\n", num);
    exit(1);
    }


    cudaMalloc((void **)&cd, num*sizeof(int));
    if(!cd)
    {printf("cannot allocated array cd of %d elements\n", num);
    exit(1);
    }

    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    //move a and b to the device
    cudaMemcpy(ad, a, num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, num*sizeof(int), cudaMemcpyHostToDevice);

    //Launch the kernel
    addvector<<<numblocks , threadsperblock>>>(ad, bd, cd, num);
    cudaDeviceSynchronize();

    //bring data back
    cudaMemcpy(c, cd, num*sizeof(int), cudaMemcpyDeviceToHost);
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("%d:%.3f\n", num, time);

    free(a);
    free(b);
    free(c);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);


}


__global__  void addvector(int * ad, int * bd, int *cd, int n)
{
   int index;

   index = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (index < n ) {
   clock_t start = clock64();
    clock_t now;
    for (;;) {
      now = clock64();
      clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= 10000000) {
        break;
      }
    }
         cd[index] = ad[index] + bd[index];
  }

}


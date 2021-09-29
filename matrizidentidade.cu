%%cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//posto da matriz identidade
#define N 13

//dimensões da grid
#define BLOCKDIMX 2
#define BLOCKDIMY 3



__global__ void MatId(int n, int* mat) {
    //i e j representam a linha e a coluna de um elemento desta matriz
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    //A matriz é representada na memória como uma array de n*n inteiros
    int id = i*n + j; //índice para acessar mat[i,j]

    if (i < n && j < n) {
      if (i == j) {
          mat[id] = 1;
      }
      else {
          mat[id] = 0;
      }      
    }
}


void printmatriz(int l, int c, int* mat) {
    //A matriz é representada com uma array de l*c inteiros
    //O elemento mat[i,j] corresponde ao índice i*c+j da array

    int i, j;

    int ic;
    for (i = 0; i < l; i++)
    {
        ic = i*c;
        for (j = 0; j < c; j++)
            printf("%d  ", mat[ic + j]);
        printf("\n");
    }
    printf("\n\n");
}


int main() {

    int* mat = (int*)malloc(N * N * sizeof(int*));

    int* dev_mat = 0;

    cudaError_t cudaStatus;

    // Alocar espaço na memória do device
    cudaStatus = cudaMalloc( ( void** ) &dev_mat , N * N * sizeof( int ) );

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    //As dimensões de uma grid são função daquelas do block
    dim3 grid( (N + BLOCKDIMX - 1)/BLOCKDIMX , (N + BLOCKDIMY - 1)/BLOCKDIMY);

    MatId<<<grid, block>>>(N, dev_mat);

    //Copia a matriz identidade de dev_mat, no device, para mat, no host
    cudaStatus = cudaMemcpy( mat , dev_mat , N * N * sizeof( int ) , cudaMemcpyDeviceToHost );
    
    printmatriz(N, N, mat);

    return 0;
}
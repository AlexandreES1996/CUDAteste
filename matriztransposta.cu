%%cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//dimensões da matriz
#define L 11
#define C 5

//dimensões da grid
#define BLOCKDIMX 3
#define BLOCKDIMY 2

__global__ void Mtrans(int l, int c, int* matin, int* matout) {
    //Preenche a matriz c*l matout com a transposta da matriz l*c matin
    //Ambas são representadas na memória com arrays de l*c inteiros

    //coluna de um elemento qualquer de matin
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    //linha de um elemento qualquer de matin
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x<C && y < L)  {
        
      //índice para matin[y, x] na array
      int idin = y*c + x;

      //matout é uma matriz c*l, e se deseja que matout[x,y] seja matin[y,x]
      int idout = x*l + y;

      matout[idout] = matin[idin];  //matout[x,y] = matin[y,x]
    }
}

void printmatriz(int l, int c, int* mat) {

    int i, j;

    for (i = 0; i < l; i++)
    {
        for (j = 0; j < c; j++)
        {
            printf("%d\t", mat[i*c + j]);
        }
        printf("\n\n");
    }
    printf("\n\n");
}

int main() {
    const int len = L*C; //número de elementos na matriz

    int* matin = (int*)malloc(len * sizeof(int*));
    int* matout = (int*)malloc(len * sizeof(int*));

    //Preenche a matriz de entrada aleatoriamente
    for(int i = 0; i < len; i++)
      matin[i] = rand();

    printf("Matriz de entrada:\n\n");
    printmatriz(L, C, matin);

    int* dev_matin = 0;
    int* dev_matout = 0;

    cudaError_t cudaStatus;
    // Alocar espaço na memória do device
    cudaStatus = cudaMalloc( ( void** ) &dev_matin , len * sizeof( int ) );
    cudaStatus = cudaMalloc( ( void** ) &dev_matout , len * sizeof( int ) );

    // Copia matin para a memória do device
    cudaStatus = cudaMemcpy( dev_matin , matin , len * sizeof( int ) , cudaMemcpyHostToDevice );

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    //As dimensões da grid são função daquelas do bloco
    dim3 grid( (C + BLOCKDIMX - 1)/BLOCKDIMX , (L + BLOCKDIMY - 1)/BLOCKDIMY);

    Mtrans<<<grid, block>>>(L, C, dev_matin, dev_matout);

    cudaStatus = cudaMemcpy( matout, dev_matout, len * sizeof( int ), cudaMemcpyDeviceToHost);
    
    printf("Matriz transposta:\n\n");
    printmatriz(C, L, matout);

    return 0;
}
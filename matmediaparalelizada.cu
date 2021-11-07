%%cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//posto da matriz
#define N 1000

#define AMP (2<<30) //valor máximo de um elemento da matriz

//dimensões do bloco
#define BLOCKDIMX 10
#define BLOCKDIMY 10

#define MODULO(X, N)  ( (X)%(N) + (N) ) %(N)


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
    printf("\n");
}

void printmatrizfloat(int l, int c, float* mat) {
    
    int i, j;

    for (i = 0; i < l; i++)
    {
        for (j = 0; j < c; j++)
        {
            printf("%g\t", mat[i*c + j]);
        }
        printf("\n\n");
    }
    printf("\n");
}


__global__ void Media(int l, int c, int* matin, float* matout) {
    //O elemento matout[i][j], da matriz de saída, torna-se a média entre o elemento correspondente da matriz de entrada
    //e seus quatro vizinhos: matin [i], matin[i-1][j], matin[i+1][j], matin[i][j-1] e matin[i][j+1].
    //Nos casos onde o índice i é negativo, ou maior ou igual ao número de linhas l,
    //ele é substituído por MODULO(i, l).
    //Semelhantemente, j = MODULO(j, c).

    //As threads deste programa são indexadas em duas dimensões.
    //Desta forma, os índices x e y da thread em execução podem ser usados
    //como os índices para a coluna e a linha do elemento da matriz a ser processado.

    //coluna de um elemento de matin:
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    //linha de um elemento de matin:
    int i = blockDim.y * blockIdx.y + threadIdx.y;

    if(j<c && i < l)  {
        
      float soma; //a média será calculada como soma/5
          
      //Matrizes são representadas da memória por arrays unidimensionais, 
      //divididas em l blocos de c elementos,
      //sendo l o número de linhas, e c o de colunas.
      //O i-ésimo bloco representa a i-ésima linha, 
      //e o c-ésimo elemento de um bloco representa o c-ésimo elemento da linha correspondente.
      //Desta forma, o elemento [i, j] de uma matriz estará na posição i*c + j da array que a representa.

      //índice para o elemento [i, j] de uma matriz:
      int id = i*c + j;
      //índices para os elementos à direita, esquerda, acima e abaixo
      int iddir = i*c + MODULO(j+1,c);
      int idesq = i*c + MODULO(j-1,c);
      int idsuper = MODULO(i-1,l)*c + j;
      int idsub = MODULO(i+1,l)*c + j;

      soma = matin[id] + matin[iddir] + matin[idesq] + matin[idsuper] + matin[idsub];

      matout[id] = soma/5;

    }
}

int main() {

    float time;
    cudaEvent_t start, stop;

    dim3 block(BLOCKDIMX, BLOCKDIMY);
    
    int len = N*N; //número de elementos na matriz

    //Aloca espaço da memória do host para as matrizes de entrada e saída
    int* matin = (int*)malloc(len * sizeof(int*));
    float* matout = (float*)malloc(len * sizeof(float*));

    //Preenche a matriz de entrada aleatoriamente
    for(int i = 0; i < len; i++)
      matin[i] = (rand())%AMP;

    //printf("\n\nMatriz de entrada:\n\n");
    //printmatriz(N, N, matin);

    //Endereços das matrizes de entrada e saída na memória do device
    int* dev_matin = 0;
    float* dev_matout = 0;


    //Começa a contar o tempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

      cudaError_t cudaStatus;

      //Alocar espaço na memória do device
      cudaStatus = cudaMalloc( ( void** ) &dev_matin , len * sizeof( int ) );
      cudaStatus = cudaMalloc( ( void** ) &dev_matout , len * sizeof( float ) );

      // Copia matin para a memória do device
      cudaStatus = cudaMemcpy( dev_matin , matin , len * sizeof( int ) , cudaMemcpyHostToDevice );

      //As dimensões da grid são função daquelas do bloco
      dim3 grid( (N + BLOCKDIMX - 1)/BLOCKDIMX , (N + BLOCKDIMY - 1)/BLOCKDIMY);
      //Estas dimensões garantem que haja uma thread para cada elemento

      //Processa a matriz
      Media<<<grid, block>>>(N, N, dev_matin, dev_matout);

      //Copia matout para o host
      cudaStatus = cudaMemcpy( matout, (float*)dev_matout, len * sizeof( float ), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

    //Para de contar o tempo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);


    //printf("Matriz média:\n");
    //printmatrizfloat(N, N, matout);

    printf("Tempo:  %3.5f ms \n", N, time);

    return 0;

}
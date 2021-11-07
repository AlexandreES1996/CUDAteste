%%cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//posto da matriz
#define N 1000

#define AMP (2<<30) //valor máximo de um elemento da matriz

#define MODULO(X, N)  ( (X)%(N) + (N) ) %(N)

//para mostrar a matriz de entrada
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

//para mostrar a matriz de saída
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
    printf("\n\n");
}

void Media(int l, int c, int* matin, float* matout) {
    //O elemento matout[i][j], da matriz de saída, torna-se a média entre o elemento correspondente da matriz de entrada
    //e seus quatro vizinhos: matin [i], matin[i-1][j], matin[i+1][j], matin[i][j-1] e matin[i][j+1].
    //Nos casos onde o índice i é negativo, ou maior ou igual ao número de linhas l,
    //ele é substituído por MODULO(i, l).
    //Semelhantemente, j = MODULO(j, c).
    //A média será calculada como soma/5
    float soma;

    //Matrizes são representadas por arrays unidimensionais, divididas em l blocos de c elementos,
    //sendo l o número de linhas, e c o de colunas.
    //O i-ésimo bloco representa a i-ésima linha, 
    //e o c-ésimo elemento de um bloco representa o c-ésimo elemento da linha correspondente.
    //Desta forma, o elemento [i, j] de uma matriz estará na posição i*c + j da array que a representa.

    //O índice para o elemento [i, j] de uma matriz é
    int id;

    //Índices para os elementos à direita, à esquerda, acima e abaixo de [i, j]
    int iddir, idesq, idsuper, idsub;

    //Dois laços aninhados são usados para iterar sobre todos os elementos da matriz.
    //A variável contadora do primeiro representa o número da linha,
    //E a do segundo, o da coluna de um elemento.
    for(int i = 0; i < c; i++)  {
        for(int j = 0; j < l; j++)  {
            id = i*c + j;
            iddir = i*c + MODULO(j+1,c);
            idesq = i*c + MODULO(j-1,c);
            idsuper = MODULO(i-1,l)*c + j;
            idsub = MODULO(i+1,l)*c + j;

            soma = matin[id] + matin[iddir] + matin[idesq] + matin[idsuper] + matin[idsub];

            matout[id] = soma/5;
        }
    } 
}


int main() {

    float time;
    cudaEvent_t start, stop;
        
    int len = N*N; //número de elementos na matriz

    int* matin = (int*)malloc(len * sizeof(int*));
    float* matout = (float*)malloc(len * sizeof(float*));

    //Preenche a matriz de entrada aleatoriamente
    for(int i = 0; i < len; i++)
      matin[i] = (rand())%AMP;

    //printf("Matriz de entrada:\n\n");
    //printmatriz(N, N, matin);

    //O tempo começa a ser contado antes do processamento da matriz.
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    Media(N, N, matin, matout);

    //O tempo para de ser contado depois do processamento.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);


    //printf("Matriz média:\n\n");
    //printmatrizfloat(N, N, matout);

    printf("Tempo de processamento:  %3.5f ms \n", time);


    return 0;

}
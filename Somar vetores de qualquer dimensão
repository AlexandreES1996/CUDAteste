%%cu
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define TAMARRAY 13
#define TAMBLOCO 4

__global__ void meu_kernel( void )
{
  printf( "ID da thread: %d\n" , threadIdx.x );
  printf( "Dim do bloco: %d\n" , blockDim.x );
  printf( "ID bloco: %d\n" , blockIdx.x );
}

__global__ void kerneladc ( int* c , int* a , int* b )
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	c[ i ] = a[ i ] + b[ i ];
}

int a[TAMARRAY];
int b[TAMARRAY];

int main( )
{

  // Preenche os vetores de entrada a e b com inteiros aleatórios
  for (int i = 0; i < TAMARRAY; i++) 
  {
      a[i] = rand();
      b[i] = rand();
  }
 
	int c[ TAMARRAY ] = { 0 };

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Aloca espaço na memória da GPU
	cudaStatus = cudaMalloc( ( void** ) &dev_c , TAMARRAY * sizeof( int ) );
	cudaStatus = cudaMalloc( ( void** ) &dev_a , TAMARRAY * sizeof( int ) );
	cudaStatus = cudaMalloc( ( void** ) &dev_b , TAMARRAY * sizeof( int ) );


	// Copia os vetores para a GPU
	cudaStatus = cudaMemcpy( dev_a , a , TAMARRAY * sizeof( int ) , cudaMemcpyHostToDevice );
	cudaStatus = cudaMemcpy( dev_b , b , TAMARRAY * sizeof( int ) , cudaMemcpyHostToDevice );

	// Executar o kernel
	kerneladc <<< ceil((float)TAMARRAY/(float)TAMBLOCO) , TAMBLOCO >>> ( dev_c , dev_a , dev_b );

	// Copia o vetor c para o host
	cudaStatus = cudaMemcpy( c , dev_c , TAMARRAY * sizeof( int ) , cudaMemcpyDeviceToHost );

  //Imprime os vetores a, b e c
  printf("\nA: (");
  for(int i = 0; i < TAMARRAY; i++)
  {
      printf("%d, ", a[i]);
  }
  printf(")");
  printf("\nB: (");
  for(int i = 0; i < TAMARRAY; i++)
  {
      printf("%d, ", b[i]);
  }
  printf(")");
  printf("\n\nC: (");
  for(int i = 0; i < TAMARRAY; i++)
  {
      printf("%u, ", c[i]);
  }
  printf(")"); 

	cudaStatus = cudaDeviceReset( );
	if ( cudaStatus != cudaSuccess )
	{
		fprintf( stderr , "cudaDeviceReset failed!" );
		return 1;
	}

	return 0;
}

%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define GRIDDIM 2
#define BLOCKDIM 3

__global__ void meu_kernel( void )
{
  printf( "ID da thread: %d\n" , threadIdx.x );
  printf( "Dim do bloco: %d\n" , blockDim.x );
  printf( "ID bloco: %d\n" , blockIdx.x );
}

// Define a variável de captura de erros
cudaError_t cudaStatus;

int main( )
{


meu_kernel <<<GRIDDIM , BLOCKDIM>>> ( );

 
// Captura o último erro ocorrido
cudaStatus = cudaGetLastError( );
if ( cudaStatus != cudaSuccess )
{
  fprintf( stderr , "meu_kernel falhou: %s\n" ,
    cudaGetErrorString( cudaStatus ) );
  goto Error;
} 
// Sincroniza a execução do kernel com a CPU
cudaStatus = cudaDeviceSynchronize( );
if ( cudaStatus != cudaSuccess )
{
  fprintf( stderr , "cudaDeviceSynchronize retornou erro %d após lançamento do kernel!\n" ,
      cudaStatus );
  goto Error;
}
Error:
// Executa a limpeza GPU
cudaStatus = cudaDeviceReset ( );
if ( cudaStatus != cudaSuccess )
{
  fprintf( stderr , "cudaDeviceReset falhou!"  );
  return 1;
}

  return 0 ;
 
}

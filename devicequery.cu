%%cu
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define KB 1024
#define MB 1048576
#define GB 1073741824

int dev, driverVersion, runtimeVersion = 0;

int main( )
{

  int nDevices;
  cudaGetDeviceCount(&nDevices);
 
  for (int dev = 0; dev < nDevices; dev++) {
      
    cudaSetDevice( dev );
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties( &deviceProp , dev );

		fprintf( stdout , "\nDevice %d: \"%s\"\n" , dev , deviceProp.name );  

		cudaDriverGetVersion( &driverVersion );
		cudaRuntimeGetVersion( &runtimeVersion );
		fprintf( stdout , "Versão do driver CUDA / Versão de runtime: %d.%d / %d.%d\n" , driverVersion / 1000 , ( driverVersion % 100 ) / 10 , runtimeVersion / 1000 , ( runtimeVersion % 100 ) / 10 );
		fprintf( stdout , "CUDA Capability Major/Minor version number: %d.%d\n" , deviceProp.major , deviceProp.minor );
		fprintf( stdout , "Número de multiprocessadores: %d \n\n" , deviceProp.multiProcessorCount );

		fprintf( stdout , "Memória constante total: %.1f kb\n" , (float)deviceProp.totalConstMem / KB);
		fprintf( stdout , "Memória compartilhada por bloco: %.1f kb\n" , (float)deviceProp.sharedMemPerBlock/KB );
		fprintf( stdout , "Memória compartilhada por multiprocessador: %.1f kb\n" , (float)deviceProp.sharedMemPerMultiprocessor/KB );
		fprintf( stdout , "Registradores por bloco: %d\n\n" , deviceProp.regsPerBlock);
		fprintf( stdout , "Memória global: %.2f gb\n" , (float)deviceProp.totalGlobalMem/GB );

		fprintf( stdout , "Tamanho de warp: %d\n" , deviceProp.warpSize);
		fprintf( stdout , "Threads por bloco: %d\n" , deviceProp.maxThreadsPerBlock);
		fprintf( stdout , "Dimensões de bloco máximas: [%d, %d, %d]\n" , deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);	
		fprintf( stdout , "Dimensões de grid máximas: [%d, %d, %d]\n" , deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);	

  }

  return 0 ;
}
/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   forward.c
**      Purpose: Foward algorithm for computing the probabilty 
**		of observing a sequence given a HMM model parameter.
**      Organization: University of Maryland
**
**      $Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $
*/
#include <stdio.h>
#include "hmm.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <iostream>
#include "forward_kernel.cu"

using namespace std;
//static char rcsid[] = "$Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $";

//#ifdef __GPU

extern "C"
void ForwardGPU(HMM *phmm, int T, int *O, double **alpha, double *pprob);

extern "C"
void ForwardGPU_Test_Header()
{

  int deviceCount = 0;
  if ( cudaGetDeviceCount(&deviceCount) != cudaSuccess )
    {
      cerr<<"Error: can't query devices on this machine"<<endl;
      return;
    }
  
  if ( deviceCount == 0 )
    {
      cerr<<"Error: can't find CUDA capable devices on this machine"<<endl;
      return;
    }
  
  for (int dev = 0; dev < deviceCount; ++dev) 
    {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      printf("%s\t ", deviceProp.name);
    }     
    
  //cout<<endl;
}


extern "C"
void ForwardGPU_Test_Driver(HMM *phmm, int T, int *O, real **alpha, real *pprob)
{

  int deviceCount = 0;
  if ( cudaGetDeviceCount(&deviceCount) != cudaSuccess )
    {
      cerr<<"Error: can't query devices on this machine"<<endl;
      return;
    }
  
  if ( deviceCount == 0 )
    {
      cerr<<"Error: can't find CUDA capable devices on this machine"<<endl;
      return;
    }
  
  printf("%d\t %d\t %d\t ", T, phmm->N, phmm->M);
  for (int dev = 0; dev < deviceCount; ++dev) 
    {
      //cudaDeviceProp deviceProp;
      //cudaGetDeviceProperties(&deviceProp, dev);
      //cout<<deviceProp.name<<"\t"<<endl;

      cudaSetDevice(dev);
      ForwardGPU(phmm, T, O, alpha, pprob);
      cudaThreadExit();
    }     
}


extern "C"
void ForwardGPU(HMM *phmm, int T, int *O, real **alpha, real *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  real sum;     /* partial sum */
 
  /* 1. Initialization */
 
  for (i = 1; i <= phmm->N; i++)
    alpha[1][i] = phmm->pi[i]* phmm->B[i][O[1]];

  /// rearrange CPU memory
  /// note that the time spent here won't be counted
  int N = phmm->N;
  int M = phmm->M;

  real *h_A, *h_B, *h_alpha; 

  h_A = (real*)malloc(sizeof(real)*N*N);
  h_B = (real*)malloc(sizeof(real)*N*M);
  h_alpha = (real*)malloc(sizeof(real)*T*N); 
  
  for ( i=0; i<N; ++i )
    {
      for ( j=0; j<N; ++j)
  	h_A[j*N + i] = phmm->A[i+1][j+1];
      for ( j=0; j<M; ++j )
  	h_B[j*N + i] = phmm->B[i+1][j+1];
    }
  
  for ( i=0; i<T; ++i )
    for ( j=0; j<N; ++j )
      {
  	h_alpha[i*N + j] = alpha[i+1][j+1];
      }
  

  /// timing starts from here, or later if you wish...
  /// initialize the data on GPU
  //printf("\tRunning GPU accelerated version\n");
  real *g_A, *g_B, *g_alpha;

  /// it turned out that 2D memory allocation is problematic
  cudaMalloc((void**)&g_A, sizeof(real)*N*N);
  cudaMalloc((void**)&g_B, sizeof(real)*N*M);
  cudaMalloc((void**)&g_alpha, sizeof(real)*N*T);
  
  cudaMemcpy(g_A, h_A, sizeof(real)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_B, h_B, sizeof(real)*N*M, cudaMemcpyHostToDevice);
  cudaMemcpy(g_alpha, h_alpha, sizeof(real)*N*T, cudaMemcpyHostToDevice);
  
  /// function signature
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, 0);
  for (t=1; t<T; ++t)
    {
      //ForwardKernelv1<<<N/32, 32>>>(O[t+1], (g_delta + (t-1)*N), (g_delta + t*N), (g_psi + t*N), g_A, g_B, N);
      //ForwardKernel<<<N/32, 32>>>(O[t+1], (g_alpha + (t-1)*N), (g_alpha + t*N), g_A, g_B, N);
      
      /// dgemv does not support in-place operation thus we need one more buffer
      //gemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy);
      
      /// Warning: be careful of the B
      //vMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff);

      cublasDgemv('n', N, N, 1.0, g_A, N, g_alpha + (t-1)*N, 1, 0.0, g_alpha + t*N, 1);
      vMulKernel<<< N/64 + (N/64==0), 32 >>>(N, g_alpha + t*N, g_B + (O[t+1]-1)*N, g_alpha + t*N);
    }
  cudaEventRecord(end, 0);

  cudaMemcpy(h_alpha, g_alpha, sizeof(real)*N*T, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(end);
  float delta_time;
  cudaEventElapsedTime(&delta_time, begin, end); 
  //fprintf(stderr, "\n CUDA time elapsed %f, %f GFLOPS\n", delta_time, T*(N+1)*N*1e-6/delta_time);
  printf("%f\t\t ", delta_time);

  /* /\* 2. Induction *\/ */
 
  /* for (t = 1; t < T; t++)  */
  /*   { */
  /*     for (j = 1; j <= phmm->N; j++)  */
  /* 	{ */
  /* 	  sum = 0.0; */
	
  /* 	  /// TODO: transpose A */
  /* 	  /// this is a dot product, consider MKL */
  /* 	  for (i = 1; i <= phmm->N; i++) */
  /* 	    sum += alpha[t][i]* (phmm->A[i][j]); */
	  
  /* 	  alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]); */
  /* 	} */
  /*   } */
 
  /* 3. Termination */
  *pprob = 0.0;
  for (i = 1; i <= phmm->N; i++)
    //*pprob += alpha[T][i];  
    *pprob += h_alpha[(T-1)*N + i-1];  
}

//#else

/* void Forward(HMM *phmm, int T, int *O, double **alpha, double *pprob) */
/* { */
/*   int     i, j;   /\* state indices *\/ */
/*   int     t;      /\* time index *\/ */
 
/*   double sum;     /\* partial sum *\/ */
 
/*   /\* 1. Initialization *\/ */
 
/*   for (i = 1; i <= phmm->N; i++) */
/*     alpha[1][i] = phmm->pi[i]* phmm->B[i][O[1]]; */
 
/*   /\* 2. Induction *\/ */
 
/*   for (t = 1; t < T; t++) { */
/*     for (j = 1; j <= phmm->N; j++) { */
/*       sum = 0.0; */

/*       /// TODO: transpose A */
/*       /// this is a dot product, consider MKL */
/*       for (i = 1; i <= phmm->N; i++) */
/* 	sum += alpha[t][i]* (phmm->A[i][j]); */
 
/*       alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]); */
/*     } */
/*   } */
 
/*   /\* 3. Termination *\/ */
/*   *pprob = 0.0; */
/*   for (i = 1; i <= phmm->N; i++) */
/*     *pprob += alpha[T][i]; */
/* } */

//#endif

/* void ForwardWithScale(HMM *phmm, int T, int *O, double **alpha,  */
/* 		      double *scale, double *pprob) */
/* /\*  pprob is the LOG probability *\/ */
/* { */
/*   int	i, j; 	/\* state indices *\/ */
/*   int	t;	/\* time index *\/ */

/*   double sum;	/\* partial sum *\/ */

/*   /\* 1. Initialization *\/ */

/*   scale[1] = 0.0;	 */
/*   for (i = 1; i <= phmm->N; i++) { */
/*     alpha[1][i] = phmm->pi[i]* (phmm->B[i][O[1]]); */
/*     scale[1] += alpha[1][i]; */
/*   } */
/*   for (i = 1; i <= phmm->N; i++)  */
/*     alpha[1][i] /= scale[1];  */
	
/*   /\* 2. Induction *\/ */

/*   for (t = 1; t <= T - 1; t++) { */
/*     scale[t+1] = 0.0; */
/*     for (j = 1; j <= phmm->N; j++) { */
/*       sum = 0.0; */
/*       for (i = 1; i <= phmm->N; i++)  */
/* 	sum += alpha[t][i]* (phmm->A[i][j]);  */

/*       alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]); */
/*       scale[t+1] += alpha[t+1][j]; */
/*     } */
/*     for (j = 1; j <= phmm->N; j++)  */
/*       alpha[t+1][j] /= scale[t+1];  */
/*   } */

/*   /\* 3. Termination *\/ */
/*   *pprob = 0.0; */

/*   for (t = 1; t <= T; t++) */
/*     *pprob += log(scale[t]); */
	
/* } */

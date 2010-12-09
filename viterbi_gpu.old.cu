/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   viterbi.c
**      Purpose: Viterbi algorithm for computing the maximum likelihood
**		state sequence and probablity of observing a sequence
**		given the model. 
**      Organization: University of Maryland
**
**      $Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $
*/

/*
**      Edited by: Philip Yang, Juncheng Chen
**      Date:   2 Dec 2010
**      File:   viterbi_gpu.c
**      Purpose: Viterbi algorithm for computing the maximum likelihood
**		state sequence and probablity of observing a sequence
**		given the model. The new code is supposed to run on
**              nVidia's CUDA capable hardware
**      Organization: University of Maryland
**
*/


#include <math.h>
#include "hmm.h"
#include "nrutil.h"
#include "cuda_runtime.h"
#include "viterbi_kernel.cu"

static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

#ifdef __cplusplus
extern "C"
{
#endif

/* extern "C" */
/* __device__ void ViterbiKernel(int Symbol, real* delta_prev, real* delta_curr, real* psi_curr, real* A, real *B, size_t N) */

//#define real float
#define VITHUGE  100000000000.0

/**
 *\brief Viterbi_GPU
 *
 *\param O observation sequence
 *\param 
 */
//#ifdef __GPU

//extern "C"
void ViterbiGPU(HMM *phmm, int T, int *O, real **delta, int **psi, 
	     int *q, real *pprob)
{
  int i, j;	/* state indices */
  int t;	/* time index */	

  //real  val;

  /* 1. Initialization  */
	
  for (i = 1; i <= phmm->N; i++) 
    {
      delta[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]);
      psi[1][i] = 0;
    }	


  /// rearrange CPU memory
  /// note that the time spent here won't be counted
  int N = phmm->N;
  int M = phmm->M;

  real *h_A, *h_B, *h_delta;
  int *h_psi;

  h_A = (real*)malloc(sizeof(real)*N*N);
  h_B = (real*)malloc(sizeof(real)*N*M);
  h_delta = (real*)malloc(sizeof(real)*T*N); 
  h_psi = (int*)malloc(sizeof(int)*T*N);
  
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
  	h_delta[i*N + j] = delta[i+1][j+1];
  	h_psi[i*N + j] = psi[i+1][j+1];
      }
  

  /// timing starts from here, or later if you wish...
  /// initialize the data on GPU
  //printf("\tRunning GPU accelerated version\n");
  real *g_A, *g_B, *g_delta;
  int *g_psi;

  /// it turned out that 2D memory allocation is problematic
  /* size_t pitch_A, pitch_B, pitch_delta, pitch_psi; */
  /* cudaMallocPitch((void**)&g_A, &pitch_A, sizeof(real)*N, sizeof(real)*N); */
  /* cudaMallocPitch((void**)&g_B, &pitch_B, sizeof(real)*N, sizeof(real)*M); */
  /* cudaMallocPitch((void**)&g_delta, &pitch_delta, sizeof(real)*T, sizeof(real)*N); */
  /* cudaMallocPitch((void**)&g_psi, &pitch_psi, sizeof(int)*T, sizeof(int)*N); */
 
  /* // transfer the memory to system */
  /* cudaMemcpy2D(g_A, pitch_A, h_A, sizeof(real)*N, sizeof(real)*N, N, cudaMemcpyHostToDevice); */
  /* cudaMemcpy2D(g_B, pitch_B, h_B, sizeof(real)*M, sizeof(real)*M, N, cudaMemcpyHostToDevice); */
  /* cudaMemcpy2D(g_delta, pitch_delta, h_delta, sizeof(real)*N, sizeof(real)*N, T, cudaMemcpyHostToDevice); */
  /* cudaMemcpy2D(g_psi, pitch_psi, h_psi, sizeof(int)*N, sizeof(int)*N, T, cudaMemcpyHostToDevice); */

  cudaMalloc((void**)&g_A, sizeof(real)*N*N);
  cudaMalloc((void**)&g_B, sizeof(real)*N*M);
  cudaMalloc((void**)&g_delta, sizeof(real)*N*T);
  cudaMalloc((void**)&g_psi, sizeof(int)*N*T);
  
  cudaMemcpy(g_A, h_A, sizeof(real)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_B, h_B, sizeof(real)*N*M, cudaMemcpyHostToDevice);
  cudaMemcpy(g_delta, h_delta, sizeof(real)*N*T, cudaMemcpyHostToDevice);
  cudaMemcpy(g_psi, h_psi, sizeof(int)*N*T, cudaMemcpyHostToDevice);
  
  /// function signature
  /// ViterbiKernel(int Symbol, real* delta_prev, real* delta_curr, real* psi_curr, real* A, real *B, size_t N)
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin, 0);

  for (t=1; t<T; ++t)
    {
      /// this kernel is a naive implementation
      //ViterbiKernel<<<N/32, 32>>>(O[t+1], (g_delta + (t-1)*N), (g_delta + t*N), (g_psi + t*N), g_A, g_B, N);

      /// each block computes a new unit
      /// 32 seems to be a sweet spot for #threads
      ViterbiKernelv1<<<N, 32>>>(O[t+1], (g_delta + (t-1)*N), (g_delta + t*N), (g_psi + t*N), g_A, g_B, N);
    }
  
  /* cudaMemcpy2D(h_delta, sizeof(real)*N, g_delta, pitch_delta, sizeof(real)*N, N, cudaMemcpyDeviceToHost); */
  /* cudaMemcpy2D(h_psi, sizeof(int)*N, g_psi, pitch_psi, sizeof(int)*N, N, cudaMemcpyDeviceToHost); */
  cudaMemcpy(h_delta, g_delta, sizeof(real)*N*T, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_psi, g_psi, sizeof(int)*N*T, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float delta_time;
  cudaEventElapsedTime(&delta_time, begin, end);
  fprintf(stderr, "Viterbi GPU Kernel, %f (ms), %f (GFLOPS)\n", delta_time, T*N*N*1e-6/delta_time);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  
  /* 3. Termination */

  *pprob = 0.0;
  q[T] = 1;
  for (i = 1; i <= N; i++) 
    {
      //if (delta[T][i] > *pprob) 
      if (h_delta[(T-1)*N + i-1] > *pprob) 
	{
	  //*pprob = delta[T][i];	
	  *pprob = h_delta[(T-1)*N + i-1];	
	  q[T] = i;
	}
    }
  

  /* 4. Path (state sequence) backtracking */

  for (t = T - 1; t >= 1; t--)
    //q[t] = psi[t+1][q[t+1]];
    q[t] = h_psi[t*N + q[t+1]-1];

  
  free(h_A);
  free(h_B);
  free(h_delta);
  free(h_psi);
  cudaFree(g_A);
  cudaFree(g_B);
  cudaFree(g_delta);
  cudaFree(g_psi);
}

//extern "C"
void ViterbiLogGPU(HMM *phmm, int T, int *O, real **delta, int **psi,
		int *q, real *pprob)
{
  /* int     i, j;   /\* state indices *\/ */
  /* int     t;      /\* time index *\/ */
 
  /* int     maxvalind; */
  /* real  maxval, val; */
  /* real  **biot; */

  /* /\* 0. Preprocessing *\/ */

  /* for (i = 1; i <= phmm->N; i++) */
  /*   phmm->pi[i] = log(phmm->pi[i]); */
  /* for (i = 1; i <= phmm->N; i++) */
  /*   for (j = 1; j <= phmm->N; j++) { */
  /*     phmm->A[i][j] = log(phmm->A[i][j]); */
  /*   } */

  /* biot = dmatrix(1, phmm->N, 1, T); */
  /* for (i = 1; i <= phmm->N; i++) */
  /*   for (t = 1; t <= T; t++) { */
  /*     biot[i][t] = log(phmm->B[i][O[t]]); */
  /*   } */
 
  /* /\* 1. Initialization  *\/ */
 
  /* for (i = 1; i <= phmm->N; i++) { */
  /*   delta[1][i] = phmm->pi[i] + biot[i][1]; */
  /*   psi[1][i] = 0; */
  /* } */
 
  /* /\* 2. Recursion *\/ */
 
  /* for (t = 2; t <= T; t++) { */
  /*   for (j = 1; j <= phmm->N; j++) { */
  /*     maxval = -VITHUGE; */
  /*     maxvalind = 1; */
  /*     for (i = 1; i <= phmm->N; i++) { */
  /* 	val = delta[t-1][i] + (phmm->A[i][j]); */
  /* 	if (val > maxval) { */
  /* 	  maxval = val; */
  /* 	  maxvalind = i; */
  /* 	} */
  /*     } */
 
  /*     delta[t][j] = maxval + biot[j][t]; */
  /*     psi[t][j] = maxvalind; */
 
  /*   } */
  /* } */
 
  /* /\* 3. Termination *\/ */
 
  /* *pprob = -VITHUGE; */
  /* q[T] = 1; */
  /* for (i = 1; i <= phmm->N; i++) { */
  /*   if (delta[T][i] > *pprob) { */
  /*     *pprob = delta[T][i]; */
  /*     q[T] = i; */
  /*   } */
  /* } */
 
 
  /* /\* 4. Path (state sequence) backtracking *\/ */

  /* for (t = T - 1; t >= 1; t--) */
  /*   q[t] = psi[t+1][q[t+1]]; */

}
 

#ifdef __cplusplus
}
#endif

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

//static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

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

void ViterbiLogGPU(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob);
void ViterbiGPU(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob);


void ViterbiGPU_Test_Header()
{

  int deviceCount = 0;
  if ( cudaGetDeviceCount(&deviceCount) != cudaSuccess )
    {
      fprintf(stderr, "Error: can't query devices on this machine\n");
      return;
    }
  
  if ( deviceCount == 0 )
    {
      fprintf(stderr,  "Error: can't query devices on this machine\n");
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


void ViterbiGPU_Test_Driver(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob)
{

  int deviceCount = 0;
  if ( cudaGetDeviceCount(&deviceCount) != cudaSuccess )
    {
      fprintf(stderr, "Error: can't query devices on this machine\n");
      return;
    }
  
  if ( deviceCount == 0 )
    {
      fprintf(stderr,  "Error: can't query devices on this machine\n");
      return;
    }
    
  for (int dev = 0; dev < deviceCount; ++dev) 
    {
      cudaSetDevice(dev);
      /* int* q = ivector(1,T); */
      /* real **delta = dmatrix(1, T, 1, phmm->N); */
      /* int **psi = imatrix(1, T, 1, phmm->N);        */
      /* real pprob; */

      ViterbiGPU(phmm, T, O, delta, psi, q, pprob);

      /* free_ivector(q, 1, T); */
      /* free_imatrix(psi, 1, T, 1, phmm->N); */
      /* free_dmatrix(delta, 1, T, 1, phmm->N); */
      cudaThreadExit();
    }     
}


real hViterbiGPU(int N, int M, real* h_A, real* h_B, real* h_pi, int T, int* query, int* q)
{
  int i;	/* state indices */
  int t;	/* time index */	

  real* h_delta = (real*)malloc(sizeof(real)*N); 
  int* h_psi = (int*)malloc(sizeof(int)*T*N);   

  for (i=0; i<N; ++i)
    h_delta[i] = h_pi[i] * h_B[(query[0]-1)*N + i];

  /// timing starts from here, or later if you wish...
  /// initialize the data on GPU
  real *g_A, *g_B, *g_delta_prev, *g_delta_curr;
  int *g_psi;

  cudaMalloc((void**)&g_A, sizeof(real)*N*N);
  cudaMalloc((void**)&g_B, sizeof(real)*N*M);
  cudaMalloc((void**)&g_delta_prev, sizeof(real)*N);
  cudaMalloc((void**)&g_delta_curr, sizeof(real)*N);
  cudaMalloc((void**)&g_psi, sizeof(int)*N*T); 
  
  cudaMemcpy(g_A, h_A, sizeof(real)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_B, h_B, sizeof(real)*N*M, cudaMemcpyHostToDevice);
  cudaMemcpy(g_delta_prev, h_delta, sizeof(real)*N, cudaMemcpyHostToDevice);
  cudaMemset(g_psi, 0, sizeof(int)*N);
  
  /// function signature
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin, 0);

  for (t=1; t<T; ++t)
    {
      /// each block computes a new row
      /// 32 seems to be a sweet spot for #threads
      //ViterbiKernelv1<<<N, 32>>>(O[t+1], (g_delta + (t-1)*N), (g_delta + t*N), (g_psi + t*N), g_A, g_B, N);
      ViterbiKernelv1<<<N, 32>>>(query[t], g_delta_prev, g_delta_curr, (g_psi + t*N), g_A, g_B, N);
      real* tmp = g_delta_prev;
      g_delta_prev = g_delta_curr;
      g_delta_curr = tmp;
    }
  
  cudaMemcpy(h_delta, g_delta_prev, sizeof(real)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_psi, g_psi, sizeof(int)*N*T, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float delta_time;
  cudaEventElapsedTime(&delta_time, begin, end);
  //fprintf(stderr, "Viterbi GPU Kernel, %f (ms), %f (GFLOPS)\n", delta_time, T*N*N*1e-6/delta_time);
  fprintf(stdout, "%f\t\t", delta_time);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);

  cudaFree(g_A);
  cudaFree(g_B);
  cudaFree(g_delta_prev);
  cudaFree(g_delta_curr);
  cudaFree(g_psi);


  /* 3. Termination */
  real prob = 0.0;
  q[T-1] = 1;
  for (i = 0; i < N; ++i) 
    {
      if (h_delta[i] > prob) 
	{
	  prob = h_delta[i];	
	  q[T-1] = i + 1;
	}
    }
  

  /* 4. Path (state sequence) backtracking */
  for (t = T - 2; t >= 0; --t)
    q[t] = h_psi[(t+1)*N + q[t+1]-1];
  
  free(h_delta);
  free(h_psi);

  return prob;
}


real hViterbiLogGPU(int N, int M, real* h_A, real* h_B, real* h_pi, int T, int* query, int* q)
{
  int i;	/* state indices */
  int t;	/* time index */	

  real* h_delta = (real*)malloc(sizeof(real)*N); 
  int* h_psi = (int*)malloc(sizeof(int)*T*N);   

  for (i=0; i<N; ++i)
    h_delta[i] = log(h_pi[i]) + log(h_B[(query[0]-1)*N + i]);

  /// timing starts from here, or later if you wish...
  /// initialize the data on GPU
  real *g_A, *g_B, *g_delta_prev, *g_delta_curr;
  int *g_psi;

  cudaMalloc((void**)&g_A, sizeof(real)*N*N);
  cudaMalloc((void**)&g_B, sizeof(real)*N*M);
  cudaMalloc((void**)&g_delta_prev, sizeof(real)*N);
  cudaMalloc((void**)&g_delta_curr, sizeof(real)*N);
  cudaMalloc((void**)&g_psi, sizeof(int)*N*T); 
  
  cudaMemcpy(g_A, h_A, sizeof(real)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(g_B, h_B, sizeof(real)*N*M, cudaMemcpyHostToDevice);
  cudaMemcpy(g_delta_prev, h_delta, sizeof(real)*N, cudaMemcpyHostToDevice);
  cudaMemset(g_psi, 0, sizeof(int)*N);
  
  /// function signature
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord(begin, 0);

  for (t=1; t<T; ++t)
    {
      /// each block computes a new row
      /// 32 seems to be a sweet spot for #threads
      //ViterbiKernelv1<<<N, 32>>>(O[t+1], (g_delta + (t-1)*N), (g_delta + t*N), (g_psi + t*N), g_A, g_B, N);
      ViterbiLogKernelv1<<<N, 32>>>(query[t], g_delta_prev, g_delta_curr, (g_psi + t*N), g_A, g_B, N);
      real* tmp = g_delta_prev;
      g_delta_prev = g_delta_curr;
      g_delta_curr = tmp;
    }
  
  cudaMemcpy(h_delta, g_delta_prev, sizeof(real)*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_psi, g_psi, sizeof(int)*N*T, cudaMemcpyDeviceToHost);
  
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float delta_time;
  cudaEventElapsedTime(&delta_time, begin, end);
  //fprintf(stderr, "Viterbi GPU Kernel, %f (ms), %f (GFLOPS)\n", delta_time, T*N*N*1e-6/delta_time);
  fprintf(stdout, "%f\t", delta_time);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);

  cudaFree(g_A);
  cudaFree(g_B);
  cudaFree(g_delta_prev);
  cudaFree(g_delta_curr);
  cudaFree(g_psi);


  /* 3. Termination */
  real prob = -VITHUGE;
  q[T-1] = 1;
  for (i = 0; i < N; ++i) 
    {
      if (h_delta[i] > prob) 
	{
	  prob = h_delta[i];	
	  q[T-1] = i + 1;
	}
    }
  

  /* 4. Path (state sequence) backtracking */
  for (t = T - 2; t >= 0; --t)
    q[t] = h_psi[(t+1)*N + q[t+1]-1];
  
  free(h_delta);
  free(h_psi);

  return prob;
}


void ViterbiGPU(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob)
{
  int i, j;	/* state indices */

  /// rearrange CPU memory
  /// note that the time spent here won't be counted
  int N = phmm->N;
  int M = phmm->M;

  real *h_A, *h_B;

  h_A = (real*)malloc(sizeof(real)*N*N);
  h_B = (real*)malloc(sizeof(real)*N*M);
  
  for ( i=0; i<N; ++i )
    {
      //h_delta[i] = phmm->pi[i+1] * (phmm->B[i+1][O[1]]);
      for ( j=0; j<N; ++j)
  	h_A[j*N + i] = phmm->A[i+1][j+1];
      for ( j=0; j<M; ++j )
  	h_B[j*N + i] = phmm->B[i+1][j+1];
    }
 
  *pprob = hViterbiGPU(N, M, h_A, h_B, phmm->pi+1, T, O+1, q+1);
    
  free(h_A);
  free(h_B);
}



//extern "C"
void ViterbiLogGPU(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob)
{

  int i, j;	/* state indices */

  /// rearrange CPU memory
  /// note that the time spent here won't be counted
  int N = phmm->N;
  int M = phmm->M;

  real *h_A, *h_B;

  h_A = (real*)malloc(sizeof(real)*N*N);
  h_B = (real*)malloc(sizeof(real)*N*M);
  
  for ( i=0; i<N; ++i )
    {
      for ( j=0; j<N; ++j)
  	h_A[j*N + i] = phmm->A[i+1][j+1];
      for ( j=0; j<M; ++j )
  	h_B[j*N + i] = phmm->B[i+1][j+1];
    }
 
  *pprob = hViterbiGPU(N, M, h_A, h_B, phmm->pi+1, T, O+1, q+1);
    
  free(h_A);
  free(h_B);
}
 

#ifdef __cplusplus
}
#endif

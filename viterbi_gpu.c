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
//#include "viterbi_kernel.cu"

static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

#define VITHUGE  100000000000.0

/**
 *\brief Viterbi_GPU
 *
 *\param O observation sequence
 *\param 
 */
#ifdef __GPU
void Viterbi(HMM *phmm, int T, int *O, double **delta, int **psi, 
	     int *q, double *pprob)
{
  int i, j;	/* state indices */
  int t;	/* time index */	

  double  val;

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

  double *h_A, *h_B, *h_delta;
  int *h_psi;

  h_A = (double*)malloc(sizeof(double)*N*N);
  h_B = (double*)malloc(sizeof(double)*N*M);
  h_delta = (double*)malloc(sizeof(double)*T*N); 
  h_psi = (int*)malloc(sizeof(int)*T*N);
  
  for ( i=0; i<N; ++i )
    {
      for ( j=0; j<N; ++j)
  	h_A[i*N + j] = phmm->A[i+1][j+1];
      for ( j=0; j<M; ++j )
  	h_B[i*M + j] = phmm->B[i+1][j+1];
    }
  
  for ( i=0; i<T; ++i )
    for ( j=0; j<N; ++j )
      {
  	h_delta[i*N + j] = delta[i+1][j+1];
  	h_psi[i*N + j] = psi[i+1][j+1];
      }
  

  /// timing starts from here, or later if you wish...
  /// initialize the data on GPU
  printf("\tRunning GPU accelerated version\n");
  double *g_A, *g_B, *g_pi, *g_delta;
  int *g_psi;

  size_t pitch_A, pitch_B, pitch_delta, pitch_psi; 
  cudaMallocPitch((void**)&g_A, &pitch_A, sizeof(double)*N, sizeof(double)*N);  
  cudaMallocPitch((void**)&g_B, &pitch_B, sizeof(double)*N, sizeof(double)*M);
  cudaMallocPitch((void**)&g_delta, &pitch_delta, sizeof(double)*T, sizeof(double)*N);
  cudaMallocPitch((void**)&g_psi, &pitch_psi, sizeof(int)*T, sizeof(int)*N);
 
  // transfer the memory to system
  cudaMemcpy2D(g_A, pitch_A, h_A, sizeof(double)*N, sizeof(double)*N, N, cudaMemcpyHostToDevice);
  cudaMemcpy2D(g_B, pitch_B, h_B, sizeof(double)*M, sizeof(double)*M, N, cudaMemcpyHostToDevice);
  cudaMemcpy2D(g_delta, pitch_delta, h_delta, sizeof(double)*N, sizeof(double)*N, T, cudaMemcpyHostToDevice);
  cudaMemcpy2D(g_psi, pitch_psi, h_psi, sizeof(double)*N, sizeof(int)*N, T, cudaMemcpyHostToDevice);
  

  ////////////////////////////////////////////////////////////////
  /// old CPU code, help only the test here
  ////////////////////////////////////////////////////////////////

  /* 2. Recursion */

  /* /// transpose  */
  /* static int is_transposed = 0; */
  /* if ( !is_transposed ) */
  /*   { */
  /*     for (i=1; i<= phmm->N; ++i) */
  /* 	for (j=1; j<= phmm->N; ++j) */
  /* 	  { */
  /* 	    double tmp = phmm->A[i][j]; */
  /* 	    phmm->A[i][j] = phmm->A[j][i]; */
  /* 	    phmm->A[j][i] = tmp; */
  /* 	  } */
  /*     is_transposed = 1; */
  /*   } */
  

  /// insert the memory operations here
  /// delta stires the best value
  /// psi stores the back tracking state
  for (t = 2; t <= T; t++) 
    {     
      //#pragma omp parallel for private(i)
      for (j = 1; j <= N; j++) 
      //for (i = 1; i <= phmm->N; i++) 
	{
	  double maxval = 0.0;
	  int maxvalidx = 1;	
	  
	  /// find the largest value
	  /// we should use the transposed matrix A
	  /// instead of the orginal version due to 

	  ///use SIMD to vectorize inner-most loop
	  for (i = 1; i <= N; i++) 
	    //for (j = 1; j <= phmm->N; j++) 
	    {
	      //val = delta[t-1][i]*(phmm->A[i][j]);
	      //val = delta[t-1][i]* h_A[(i-1)*N + j-1];
	      val = h_delta[(t-2)*N + i-1]* h_A[(i-1)*N + j-1];

	      //val = delta[t-1][i]*(phmm->A[j][i]);

	      if (val > maxval) 
		{
		  maxval = val;	
		  maxvalidx = i;	
		}
	    }
	  
	  //delta[t][j] = maxval*(phmm->B[j][O[t]]);
	  h_delta[(t-1)*N + j-1] = maxval*( h_B[(j-1)*M + (O[t]-1)] );

	  //psi[t][j] = maxvalidx; 	  
	  h_psi[(t-1)*N + j-1] = maxvalidx; 	  
	}
    }
  
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

#else 

void Viterbi(HMM *phmm, int T, int *O, double **delta, int **psi, 
	     int *q, double *pprob)
{
  int 	i, j;	/* state indices */
  int  	t;	/* time index */	

  int	maxvalidx;
  double	maxval, val;

  printf("\t\tRunning CPU accelerated version\n");

  /* 1. Initialization  */
	
  for (i = 1; i <= phmm->N; i++) {
    delta[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]);
    psi[1][i] = 0;
  }	

  /* 2. Recursion */
	
  for (t = 2; t <= T; t++) 
    {
      for (j = 1; j <= phmm->N; j++) 
	{
	  maxval = 0.0;
	  maxvalidx = 1;	
	  for (i = 1; i <= phmm->N; i++) 
	    {
	      val = delta[t-1][i]*(phmm->A[i][j]);
	      if (val > maxval) 
		{
		  maxval = val;	
		  maxvalidx = i;	
		}
	    }
	  
	  delta[t][j] = maxval*(phmm->B[j][O[t]]);
	  psi[t][j] = maxvalidx; 	  
	}
    }
  
  /* 3. Termination */

  *pprob = 0.0;
  q[T] = 1;
  for (i = 1; i <= phmm->N; i++) 
    {
      if (delta[T][i] > *pprob) 
	{
	  *pprob = delta[T][i];	
	  q[T] = i;
	}
    }
  

  /* 4. Path (state sequence) backtracking */

  for (t = T - 1; t >= 1; t--)
    q[t] = psi[t+1][q[t+1]];

}

#endif


void ViterbiLog(HMM *phmm, int T, int *O, double **delta, int **psi,
		int *q, double *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  int     maxvalind;
  double  maxval, val;
  double  **biot;

  /* 0. Preprocessing */

  for (i = 1; i <= phmm->N; i++) 
    phmm->pi[i] = log(phmm->pi[i]);
  for (i = 1; i <= phmm->N; i++) 
    for (j = 1; j <= phmm->N; j++) {
      phmm->A[i][j] = log(phmm->A[i][j]);
    }

  biot = dmatrix(1, phmm->N, 1, T);
  for (i = 1; i <= phmm->N; i++) 
    for (t = 1; t <= T; t++) {
      biot[i][t] = log(phmm->B[i][O[t]]);
    }
 
  /* 1. Initialization  */
 
  for (i = 1; i <= phmm->N; i++) {
    delta[1][i] = phmm->pi[i] + biot[i][1];
    psi[1][i] = 0;
  }
 
  /* 2. Recursion */
 
  for (t = 2; t <= T; t++) {
    for (j = 1; j <= phmm->N; j++) {
      maxval = -VITHUGE;
      maxvalind = 1;
      for (i = 1; i <= phmm->N; i++) {
	val = delta[t-1][i] + (phmm->A[i][j]);
	if (val > maxval) {
	  maxval = val;
	  maxvalind = i;
	}
      }
 
      delta[t][j] = maxval + biot[j][t]; 
      psi[t][j] = maxvalind;
 
    }
  }
 
  /* 3. Termination */
 
  *pprob = -VITHUGE;
  q[T] = 1;
  for (i = 1; i <= phmm->N; i++) {
    if (delta[T][i] > *pprob) {
      *pprob = delta[T][i];
      q[T] = i;
    }
  }
 
 
  /* 4. Path (state sequence) backtracking */

  for (t = T - 1; t >= 1; t--)
    q[t] = psi[t+1][q[t+1]];

}
 


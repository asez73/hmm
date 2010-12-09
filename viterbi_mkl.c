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

#include <math.h>
#include "hmm.h"
#include <omp.h>
#include "nrutil.h"

static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

#define VITHUGE  100000000000.0


void Viterbi(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob);

void ViterbiMKL_Test_Driver(HMM *phmm, int T, int *O, real **delta, int **psi, int *q, real *pprob)
{
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      omp_set_num_threads(p);
      Viterbi(phmm, T, O, delta, psi, q, pprob);

      //cout<<deviceProp.name<<"\t";
      //printf("%d threads\t ", p);
    }             
}


/// simplified version
real hViterbi(int N, int M, real* A, real* B, real* pi, int T, int *query, int *q)
{
  int 	i, j;	/* state indices */
  int  	t;	/* time index */	

  int	maxvalidx;
  real	maxval;
    
  //#pragma omp parallel 
  int nproc = omp_get_max_threads();

  fprintf(stderr, "%d omp threads\n", nproc);
  
  //real* buff = (real*)malloc(N*sizeof(real));
  real* buff = (real*)malloc(N*sizeof(real)*nproc);
  
  real* delta_prev = (real*)malloc(N*sizeof(real));
  real* delta_curr = (real*)malloc(N*sizeof(real));
  int* psi = (int*)malloc(N*T*sizeof(int));

  /* 1. Initialization  */
  for (i = 0; i < N; ++i)
    delta_prev[i] = pi[i] * B[(query[0]-1)*N + i];
  memset((void*)psi, 0, sizeof(int)*N);

  
  /* 2. Recursion */
  double delta_time = wallclock();
  for (t = 1; t < T; ++t)
    {
#pragma omp parallel for
      for (j = 0; j < N; ++j)
  	{	  
	  int tid = omp_get_thread_num();
	  real* local_buff = buff + tid * N;
  	  vdMul(N, delta_prev, (A + j*N), local_buff);
  	  int one = 1;
  	  maxvalidx = idamax(&N, local_buff, &one);

  	  delta_curr[j] = local_buff[maxvalidx-1] * B[(query[t]-1)*N + j];
  	  psi[t*N + j] = maxvalidx;
  	}

      real* tmp = delta_prev;
      delta_prev = delta_curr;
      delta_curr = tmp;
    }
  delta_time = wallclock() - delta_time;
  printf("%f\t ", delta_time);

  
  /* 3. Termination */
  real prob = 0.0;
  q[T-1] = 1;
  for (i = 0; i < N; ++i)
    {
      if (delta_prev[i] > prob)
  	{
  	  prob = delta_prev[i];
  	  q[T-1] = i+1;
  	}
    }

  //fprintf(stderr, "state sequence\n\t%d", q[T-1]);
  for (t = T - 2; t >= 0; --t)
    q[t] = psi[(t+1)*N + q[t+1]-1];
    //fprintf(stderr, "-> %d",q[t] = psi[(t+1)*N + q[t+1]-1]);
  //fprintf(stderr, "\n");

  
  free(buff);
  free(delta_prev);
  free(delta_curr);
  free(psi);

  return prob;
}


real hViterbiLog(int N, int M, real* A, real* B, real* pi, int T, int *query, int *q)
{
  int 	i, j;	/* state indices */
  int  	t;	/* time index */	

  int	maxvalidx;
  real	maxval;
  
  real* buff = (real*)malloc(N*sizeof(real));
  real* delta_prev = (real*)malloc(N*sizeof(real));
  real* delta_curr = (real*)malloc(N*sizeof(real));
  int* psi = (int*)malloc(N*T*sizeof(int));

  /* 1. Initialization  */
  for (i = 0; i < N; ++i)
    delta_prev[i] = pi[i] + B[(query[0]-1)*N + i];
  memset((void*)psi, 0, sizeof(int)*N);
  
  /* 2. Recursion */
  double delta_time = wallclock();
  for (t = 1; t < T; ++t)
    {
      for (j = 0; j < N; ++j)
  	{
	  vdAdd(N, delta_prev, (A + j*N), buff);
  	  int one = 1;
  	  maxvalidx = idamax(&N, buff, &one);

  	  delta_curr[j] = buff[maxvalidx-1] + B[(query[t]-1)*N + j];
  	  psi[t*N + j] = maxvalidx;
  	}

      real* tmp = delta_prev;
      delta_prev = delta_curr;
      delta_curr = tmp;
    }
  delta_time = wallclock() - delta_time;
  printf("%f\t ", delta_time);
  

  
  /* 3. Termination */
  real prob = -VITHUGE;
  q[T-1] = 1;
  for (i = 0; i < N; ++i)
    {
      if (delta_prev[i] > prob)
  	{
  	  prob = delta_prev[i];
  	  q[T-1] = i+1;
  	}
    }

  //fprintf(stderr, "state sequence\n\t%d", q[T-1]);
  for (t = T - 2; t >= 0; --t)
    q[t] = psi[(t+1)*N + q[t+1]-1];
    //fprintf(stderr, "-> %d",q[t] = psi[(t+1)*N + q[t+1]-1]);
  //fprintf(stderr, "\n");

  
  free(buff);
  free(delta_prev);
  free(delta_curr);
  free(psi);

  return prob;
}




void Viterbi(HMM *phmm, int T, int *O, real **delta, int **psi, 
	     int *q, real *pprob)
{
  real *A, *B, *pi;
  //int* q = (int*)malloc(T*sizeof(int));
  data_format(phmm, &A, &B, &pi);

  //double delta_time = wallclock();
  *pprob = hViterbi(phmm->N, phmm->M, A, B, pi, T, O+1, q+1);
  //delta_time = wallclock() - delta_time;
  //printf("%f\t ", delta_time);

  //printf("dsadwjadwa\n");
  free(A);
  free(B);
  free(pi);
}



void ViterbiLog(HMM *phmm, int T, int *O, real **delta, int **psi,
		int *q, real *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  int     maxvalind;
  real  maxval, val;
  real  **biot;


  real *A, *B, *pi;
  //int* q = (int*)malloc(T*sizeof(int));
  data_format_log(phmm, &A, &B, &pi);  

  *pprob = hViterbiLog(phmm->N, phmm->M, A, B, pi, T, O+1, q+1);

  free(A);
  free(B);
  free(pi);


  /* /\* 0. Preprocessing *\/ */

  /* for (i = 1; i <= phmm->N; i++)  */
  /*   phmm->pi[i] = log(phmm->pi[i]); */
  /* for (i = 1; i <= phmm->N; i++)  */
  /*   for (j = 1; j <= phmm->N; j++) { */
  /*     phmm->A[i][j] = log(phmm->A[i][j]); */
  /*   } */

  /* biot = dmatrix(1, phmm->N, 1, T); */
  /* for (i = 1; i <= phmm->N; i++)  */
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
 
  /*     delta[t][j] = maxval + biot[j][t];  */
  /*     psi[t][j] = maxvalind;  */
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
 


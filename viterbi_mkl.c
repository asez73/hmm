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
#include "nrutil.h"
static char rcsid[] = "$Id: viterbi.c,v 1.1 1999/05/06 05:25:37 kanungo Exp kanungo $";

#define VITHUGE  100000000000.0

void Viterbi(HMM *phmm, int T, int *O, real **delta, int **psi, 
	     int *q, real *pprob)
{
  int 	i, j;	/* state indices */
  int  	t;	/* time index */	

  int	maxvalidx;
  real	maxval, val;
  

  int N = phmm->N;
  int M = phmm->M;
  real* A = (real*)malloc(N*N*sizeof(real));
  //real* B = (real*)malloc(N*M*sizeof(real));
  real* buff = (real*)malloc(N*sizeof(real));
  real* delta_prev = (real*)malloc(N*sizeof(real));
  real* delta_curr = (real*)malloc(N*sizeof(real));

  /* 1. Initialization  */
	
  for (i = 1; i <= phmm->N; i++) 
    {
      delta_prev[i-1] = delta[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]);
      psi[1][i] = 0;
    }	


  /* for ( i=0; i<N; ++i ) */
  /*   buff[i] = alpha[1][i+1]; */
  
  // marshal the data
  for (i=0; i<N; ++i)
    for (j=0; j<N; ++j)
      A[j*N+i] = phmm->A[i+1][j+1];
  
  /* 2. Recursion */
	
  for (t = 2; t <= T; t++) 
    {
      for (j = 1; j <= N; j++) 
	{
	  /// old code
	  /* maxval = 0.0; */
	  /* maxvalidx = 1; */
	  /* for (i = 1; i <= phmm->N; i++) */
	  /*   { */
	  /*     //val = delta[t-1][i]*(phmm->A[i][j]); */
	  /*     //val = delta[t-1][i] * A[(j-1)*N + i-1]; */
	  /*     val = delta_prev[i-1] * A[(j-1)*N + i-1]; */

	  /*     if (val > maxval) */
	  /* 	{ */
	  /* 	  maxval = val; */
	  /* 	  maxvalidx = i; */
	  /* 	} */
	  /*   } */

	  /* printf("maxval: %f, maxidx %d\n", maxval, maxvalidx); */

	  /* delta_curr[j-1] = maxval*(phmm->B[j][O[t]]); */
	  /* psi[t][j] = maxvalidx; */

	  /// new code
	  
	  vdMul(N, delta_prev, (A + (j-1)*N), buff);
	  int one = 1;
	  maxvalidx = idamax(&N, buff, &one);
	  
	  /* for (i=0; i<N; ++i) */
	  /*   printf("%f ", buff[i]); */
	  /* printf("\n"); */
	  //printf("maxval: %f, maxidx %d\n\n", buff[maxvalidx-1], maxvalidx); 
	 	  
	  delta_curr[j-1] = buff[maxvalidx-1] * (phmm->B[j][O[t]]); 
	  psi[t][j] = maxvalidx; 
	}

      real* tmp = delta_prev;
      delta_prev = delta_curr;
      delta_curr = tmp;
     	  
    }
  
  /* 3. Termination */

  *pprob = 0.0;
  q[T] = 1;
  for (i = 1; i <= phmm->N; i++)
    {
      //if (delta[T][i] > *pprob) 
      if (delta_prev[i-1] > *pprob) 
	{
	  //*pprob = delta[T][i];	
	  *pprob = delta_prev[i-1];	
	  q[T] = i;
	}
    }
  
  /* 4. Path (state sequence) backtracking */
  
  for (t = T - 1; t >= 1; t--)
    q[t] = psi[t+1][q[t+1]];

}


void ViterbiLog(HMM *phmm, int T, int *O, real **delta, int **psi,
		int *q, real *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  int     maxvalind;
  real  maxval, val;
  real  **biot;

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
 


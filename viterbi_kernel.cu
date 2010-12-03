/*
**      Edited by: Philip Yang, Juncheng Chen
**      Date:   2 Dec 2010
**      File:   viterbi_kernel.cu
**      Purpose: Viterbi algorithm for computing the maximum likelihood
**		state sequence and probablity of observing a sequence
**		given the model. The new code is supposed to run on
**              nVidia's CUDA capable hardware
**      Organization: University of Maryland
**
*/

#include "hmm.h"


/// this kernel deals with 
void viterbi_kernel_small(HMM *phmm, int T, int *O, double **delta, int **psi, 
		    int *q, double *pprob)
{

  

  /* int 	i, j;	/\* state indices *\/ */
  /* int  	t;	/\* time index *\/	 */

  /* int	maxvalind; */
  /* double	maxval, val; */

  /* /\* 1. Initialization  *\/ */
	
  /* for (i = 1; i <= phmm->N; i++) { */
  /*   delta[1][i] = phmm->pi[i] * (phmm->B[i][O[1]]); */
  /*   psi[1][i] = 0; */
  /* }	 */

  /* /\* 2. Recursion *\/ */
	
  /* for (t = 2; t <= T; t++) { */
  /*   for (j = 1; j <= phmm->N; j++) { */
  /*     maxval = 0.0; */
  /*     maxvalind = 1;	 */
  /*     for (i = 1; i <= phmm->N; i++) { */
  /* 	val = delta[t-1][i]*(phmm->A[i][j]); */
  /* 	if (val > maxval) { */
  /* 	  maxval = val;	 */
  /* 	  maxvalind = i;	 */
  /* 	} */
  /*     } */
			
  /*     delta[t][j] = maxval*(phmm->B[j][O[t]]); */
  /*     psi[t][j] = maxvalind;  */

  /*   } */
  /* } */

  /* /\* 3. Termination *\/ */

  /* *pprob = 0.0; */
  /* q[T] = 1; */
  /* for (i = 1; i <= phmm->N; i++) { */
  /*   if (delta[T][i] > *pprob) { */
  /*     *pprob = delta[T][i];	 */
  /*     q[T] = i; */
  /*   } */
  /* } */

  /* /\* 4. Path (state sequence) backtracking *\/ */

  /* for (t = T - 1; t >= 1; t--) */
  /*   q[t] = psi[t+1][q[t+1]]; */

}


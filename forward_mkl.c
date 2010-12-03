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
static char rcsid[] = "$Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $";

void Forward(HMM *phmm, int T, int *O, double **alpha, double *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  double sum;     /* partial sum */
 
  /* 1. Initialization */
 
  for (i = 1; i <= phmm->N; i++)
    alpha[1][i] = phmm->pi[i]* phmm->B[i][O[1]];
 
  /* 2. Induction */
  int N = phmm->N;
  int M = phmm->M;
  double* A = (double*)malloc(N*N*sizeof(double));
  double* B = (double*)malloc(N*M*sizeof(double));
  double* buff = (double*)malloc(N*sizeof(double));
  double* buff_tmp = (double*)malloc(N*sizeof(double));

  for ( i=0; i<N; ++i )
    buff[i] = alpha[1][i+1];
  
  // marshal the data
  for (i=0; i<N; ++i)
    {
      for (j=0; j<N; ++j)
	{
	  A[i*N+j] = phmm->A[i+1][j+1];
	}
      for (j=0; j<M; ++j)
	{
	  B[j*N+i] = phmm->B[i+1][j+1];
	}
    }

  char trans = 'n';
  double alp = 1, beta = 0;
  int incx = 1, incy = 1;
  for (t = 1; t < T; t++) 
    {     
      dgemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy);
      
      /// Warning: be careful of the B
      vdMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff);              
      
      /* for (j = 1; j <= phmm->N; j++)  */
      /* 	{ */
      /* 	  sum = 0.0; */
	  
      /* 	  //TODO: transpose A */
      /* 	  //this is a dot product, consider MKL */
      /* 	  for (i = 1; i <= phmm->N; i++) */
      /* 	    sum += alpha[t][i]* (phmm->A[i][j]); */

      /* 	  alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]);	   */
      /* 	  //alpha[t+1][j] = sum; */
	  
      /* 	  /\* for (i = 0; i < N; i++) *\/ */
      /* 	  /\*   sum += buff[j-1]* (A[i*N+j-1]); *\/ */
	  
      /* 	  /\* alpha[t+1][j] = sum*(B[N*(O[t+1] -1) + j-1]);      	 *\/ */
      /* 	} */
      
      /* for ( i=0; i<N; ++i ) */
      /* 	{ */
      /* 	  if ( alpha[t+1][i+1] != buff[i] ) */
      /* 	    printf("%f %f\n", alpha[t+1][i+1], buff[i] );	   */
      /* 	} */
      //break;

      /* for(int j=0; j < N; ++j) */
      /* 	printf("%lf ", buff[j]); */
      /* printf("\n"); */
      /* getchar(); */
    }
  
  /* 3. Termination */
  *pprob = 0.0;
  /* for (i = 1; i <= phmm->N; i++) */
  /*   *pprob += alpha[T][i]; */
  for (i = 0; i < N; i++)
    *pprob += buff[i];
  //printf("%f\n", *pprob += buff[i]);

  free(A);
  free(B);  
  free(buff);
  free(buff_tmp);
}

void ForwardWithScale(HMM *phmm, int T, int *O, double **alpha, 
		      double *scale, double *pprob)
/*  pprob is the LOG probability */
{
  int	i, j; 	/* state indices */
  int	t;	/* time index */

  double sum;	/* partial sum */

  /* 1. Initialization */

  scale[1] = 0.0;	
  for (i = 1; i <= phmm->N; i++) {
    alpha[1][i] = phmm->pi[i]* (phmm->B[i][O[1]]);
    scale[1] += alpha[1][i];
  }
  for (i = 1; i <= phmm->N; i++) 
    alpha[1][i] /= scale[1]; 
	
  /* 2. Induction */

  for (t = 1; t <= T - 1; t++) {
    scale[t+1] = 0.0;
    for (j = 1; j <= phmm->N; j++) {
      sum = 0.0;
      for (i = 1; i <= phmm->N; i++) 
	sum += alpha[t][i]* (phmm->A[i][j]); 

      alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]);
      scale[t+1] += alpha[t+1][j];
    }
    for (j = 1; j <= phmm->N; j++) 
      alpha[t+1][j] /= scale[t+1]; 
  }

  /* 3. Termination */
  *pprob = 0.0;

  for (t = 1; t <= T; t++)
    *pprob += log(scale[t]);
	
}

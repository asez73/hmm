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
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <omp.h>
#include "hmm.h"
#include "nrutil.h"

//static char rcsid[] = "$Id: forward.c,v 1.2 1998/02/19 12:42:31 kanungo Exp kanungo $";

void Forward(HMM *phmm, int T, int *O, real **alpha, real *pprob);
void ForwardWithScale(HMM *phmm, int T, int *O, real **alpha, real* scale, real *pprob);

void ForwardMKL_Test_Driver(HMM *phmm, int T, int *O, real **alpha, real *pprob)
{
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      omp_set_num_threads(p);
      Forward(phmm, T, O, alpha, pprob);

      //cout<<deviceProp.name<<"\t";
      //printf("%d threads\t ", p);
    }             
}


void ForwardScaleMKL_Test_Driver(HMM *phmm, int T, int *O, real **alpha, real* scale, real *pprob)
{
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      omp_set_num_threads(p);      
      ForwardWithScale(phmm, T, O, alpha, scale, pprob);      
      //cout<<deviceProp.name<<"\t";
      //printf("%d threads\t ", p);
    }             
}


void Forward(HMM *phmm, int T, int *O, real **alpha, real *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  real sum;     /* partial sum */
 
  /* 1. Initialization */
 
  for (i = 1; i <= phmm->N; i++)
    alpha[1][i] = phmm->pi[i]* phmm->B[i][O[1]];
 
  /* 2. Induction */
  int N = phmm->N;
  int M = phmm->M;
  real* A = (real*)malloc(N*N*sizeof(real));
  real* B = (real*)malloc(N*M*sizeof(real));
  real* buff = (real*)malloc(N*sizeof(real));
  real* buff_tmp = (real*)malloc(N*sizeof(real));

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
  real alp = 1, beta = 0;
  int incx = 1, incy = 1;

  double delta_time = wallclock();
  for (t = 1; t < T; t++) 
    {     

      /* /// dgemv does not support in-place operation thus we need one more buffer */
      /* sgemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy); */
      
      /* /// Warning: be careful of the B */
      /* vsMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff); */

      /// dgemv does not support in-place operation thus we need one more buffer
      gemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy);
      
      /// Warning: be careful of the B
      vMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff);

      
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
  delta_time = wallclock() - delta_time;
  printf("%f\t ", delta_time);

  
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



void ForwardWithScale(HMM *phmm, int T, int *O, real **alpha, real* scale, real *pprob)
{
  int     i, j;   /* state indices */
  int     t;      /* time index */
 
  real sum;     /* partial sum */
 
  /* 1. Initialization */
   for (i = 1; i <= phmm->N; i++)
    alpha[1][i] = phmm->pi[i]* phmm->B[i][O[1]];
 
  /* 2. Induction */
  int N = phmm->N;
  int M = phmm->M;
  real* A = (real*)malloc(N*N*sizeof(real));
  real* B = (real*)malloc(N*M*sizeof(real));
  real* buff = (real*)malloc(N*sizeof(real));
  real* buff_tmp = (real*)malloc(N*sizeof(real));
  real* scale_buff = (real*)malloc(N*sizeof(real));

  memset((void*)scale_buff, 0, sizeof(real)*N);
  for ( i=0; i<N; ++i )
    scale_buff[0] += (buff[i] = alpha[1][i+1]);
  for ( i=0; i<N; ++i )
    buff[i] /= scale_buff[0]; 

  /// marshal the data
  /// the time taken is marginal 
  /// compared with the time consumed in computation
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
  real alp = 1, beta = 0;
  int incx = 1, incy = 1;

  double delta_time = wallclock();
  for (t = 1; t < T; t++) 
    {     

      /* /// dgemv does not support in-place operation thus we need one more buffer */
      /* sgemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy); */
      
      /* /// Warning: be careful of the B */
      /* vsMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff); */

      /// dgemv does not support in-place operation thus we need one more buffer
      gemv(&trans, &N, &N, &alp, A, &N, buff, &incx, &beta, buff_tmp, &incy);
      
      /// Warning: be careful of the B
      vMul(N, buff_tmp, &(B[(O[t+1]-1)*N]), buff);

      /// cumulate the scaling factor
      /// Warning: remember to include mkl.h so that the right C functions are called 
      int one = 1;
      scale_buff[t] = asum(&N, buff, &one);
      real factor = 1.0 / scale_buff[t] - 1;
      
      /// scale the alpha vector
      axpy(&N, &factor, buff, &one, buff, &one);
    }
  delta_time = wallclock() - delta_time;
  printf("%f\t ", delta_time);


  /* 3. Termination */
  *pprob = 0.0;

  for (t = 0; t < T; t++)
    *pprob += log(scale_buff[t]);

  free(A);
  free(B);  
  free(buff);
  free(buff_tmp);
}



/* void ForwardWithScale(HMM *phmm, int T, int *O, real **alpha,  */
/* 		      real *scale, real *pprob) */
/* /\*  pprob is the LOG probability *\/ */
/* { */
/*   int	i, j; 	/\* state indices *\/ */
/*   int	t;	/\* time index *\/ */

/*   real sum;	/\* partial sum *\/ */

/*   /\* 1. Initialization *\/ */

/*   scale[1] = 0.0;	 */
/*   for (i = 1; i <= phmm->N; i++)  */
/*     { */
/*       alpha[1][i] = phmm->pi[i]* (phmm->B[i][O[1]]); */
/*       scale[1] += alpha[1][i]; */
/*     } */
/*   for (i = 1; i <= phmm->N; i++)  */
/*     alpha[1][i] /= scale[1];  */
	
/*   /\* 2. Induction *\/ */

/*   for (t = 1; t <= T - 1; t++)  */
/*     { */
/*       scale[t+1] = 0.0; */

/*       for (j = 1; j <= phmm->N; j++)  */
/* 	{ */
/* 	  sum = 0.0; */
/* 	  for (i = 1; i <= phmm->N; i++)  */
/* 	    sum += alpha[t][i]* (phmm->A[i][j]);  */
	  
/* 	  alpha[t+1][j] = sum*(phmm->B[j][O[t+1]]); */
/* 	  scale[t+1] += alpha[t+1][j]; */
/* 	} */

/*       for (j = 1; j <= phmm->N; j++)  */
/* 	alpha[t+1][j] /= scale[t+1];  */
/*     } */
  
/*   /\* 3. Termination *\/ */
/*   *pprob = 0.0; */

/*   for (t = 1; t <= T; t++) */
/*     *pprob += log(scale[t]); */
	
/* } */

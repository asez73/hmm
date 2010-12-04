#include <stdio.h>
#include <stdlib.h>
#include "hmm.h" 


/// generate a random HMM model
void GenRandHMM(HMM *phmm)		
{
  int i, j;
  int N = phmm->N;
  int M = phmm->M;
  //real *arr = (real*)malloc(sizeof(real)*N);
  
  srand(time(NULL));
  
  printf("Generating Random HMM\n");
  printf("N: %d\nM: %d\n", N, M);

  //printf("A:\n");
  for (i=1; i<=N; ++i)
    {
      real norm = 0.0;
      for (j=1; j<=N; ++j)
	norm += (phmm->A[i][j] = (1 + (real)rand()/RAND_MAX));      
      for (j=1; j<=N; ++j)
	phmm->A[i][j] /= norm;
	//printf("%f ", phmm->A[i][j] /= norm);
      //printf("\n");
    }

  //printf("B:\n");
  for (i=1; i<=N; ++i)
    {
      real norm = 0.0;
      for (j=1; j<=M; ++j)
	norm += (phmm->B[i][j] = (1 + (real)rand()/RAND_MAX));
      for (j=1; j<=M; ++j)
	//printf("%f ", phmm->B[i][j] /= norm);
	phmm->B[i][j] /= norm;

    }

  printf("Pi:\n");
  real norm = 0.0;
  for (i=1; i<=N; ++i)
    norm += (phmm->pi[i] = (1 + (real)rand()/RAND_MAX));
  for (i=1; i<=N; ++i)
    phmm->pi[i] / norm;
    //printf("%f ", phmm->pi[i] / norm);
  //printf("\n");
}


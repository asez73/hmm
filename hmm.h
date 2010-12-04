/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   hmm.h
**      Purpose: datastructures used for HMM. 
**      Organization: University of Maryland
**
**	Update:
**	Author: Tapas Kanungo
**	Purpose: include <math.h>. Not including this was
**		creating a problem with forward.c
**      $Id: hmm.h,v 1.9 1999/05/02 18:38:11 kanungo Exp kanungo $
*/

#ifdef __DOUBLE
#define real double
#define vMul vdMul
#define gemv dgemv
#elif defined(__SINGLE)
#define real float
#define vMul vsMul
#define gemv sgemv
#else
#error "please specify the precision"
#endif

#ifndef __HMM_HEADER__
#define __HMM_HEADER__

#ifdef __cplusplus
extern "C"
{
#endif


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
  int N;		/* number of states;  Q={1,2,...,N} */
  int M; 		/* number of observation symbols; V={1,2,...,M}*/
  real	**A;	/* A[1..N][1..N]. a[i][j] is the transition prob
			   of going from state i at time t to state j
			   at time t+1 */
  real	**B;	/* B[1..N][1..M]. b[j][k] is the probability of
			   of observing symbol k in state j */
  real	*pi;	/* pi[1..N] pi[i] is the initial state distribution. */
} HMM;

void ReadHMM(FILE *fp, HMM *phmm);
void PrintHMM(FILE *fp, HMM *phmm);
void InitHMM(HMM *phmm, int N, int M, int seed);
void CopyHMM(HMM *phmm1, HMM *phmm2);
void FreeHMM(HMM *phmm);

void ReadSequence(FILE *fp, int *pT, int **pO);
void PrintSequence(FILE *fp, int T, int *O);
void GenSequenceArray(HMM *phmm, int seed, int T, int *O, int *q);
int GenInitalState(HMM *phmm);
int GenNextState(HMM *phmm, int q_t);
int GenSymbol(HMM *phmm, int q_t);

  
/// generate a random HMM model
void GenRandHMM(HMM *phmm);

 
void Forward(HMM *phmm, int T, int *O, real **alpha, real *pprob);
void ForwardWithScale(HMM *phmm, int T, int *O, real **alpha,
		      real *scale, real *pprob);
void Backward(HMM *phmm, int T, int *O, real **beta, real *pprob);
void BackwardWithScale(HMM *phmm, int T, int *O, real **beta,
		       real *scale, real *pprob);
void BaumWelch(HMM *phmm, int T, int *O, real **alpha, real **beta,
	       real **gamma, int *niter, 
	       real *plogprobinit, real *plogprobfinal);

real *** AllocXi(int T, int N);
void FreeXi(real *** xi, int T, int N);
void ComputeGamma(HMM *phmm, int T, real **alpha, real **beta,
		  real **gamma);
void ComputeXi(HMM* phmm, int T, int *O, real **alpha, real **beta,
	       real ***xi);
void Viterbi(HMM *phmm, int T, int *O, real **delta, int **psi,
	     int *q, real *pprob);
void ViterbiLog(HMM *phmm, int T, int *O, real **delta, int **psi,
		int *q, real *pprob);

/* random number generator related functions*/

int hmmgetseed(void);
void hmmsetseed(int seed);
real hmmgetrand(void);
 
#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))
 

#ifdef __cplusplus
}
#endif


#endif

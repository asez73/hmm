/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   testvit.c
**      Purpose: driver for testing the Viterbi code.
**      Organization: University of Maryland
**
**	Update:
**	Author:	Tapas Kanungo
**	Purpose: run both viterbi with probabilities and 
**		viterbi with log, change output etc.
**      $Id: testvit.c,v 1.3 1998/02/23 07:39:07 kanungo Exp kanungo $
*/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include "nrutil.h"
#include "hmm.h"

static char rcsid[] = "$Id: testvit.c,v 1.3 1998/02/23 07:39:07 kanungo Exp kanungo $";

double wallclock(void)
{
  struct timeval tv;                                                                                                
  struct timezone tz;                                                                                               
  double t;                                                                                                         

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec * 1000;                                                                                     
  t += ((double)tv.tv_usec) * 0.001;   

  return t;
}

int main (int argc, char **argv)
{
  int 	t, T; 
  HMM  	hmm;
  int	*O;	/* observation sequence O[1..T] */
  int	*q;	/* state sequence q[1..T] */
  real **delta;
  int	**psi;
  real 	proba, logproba; 
  FILE	*fp;

  if (argc != 4) {
    printf("Usage error \n");
    //printf("Usage: testvit <model.hmm> <obs.seq> \n");
    printf("Usage: testvit <sequence>  <num_states> <num_symbols> \n");
    exit (1);
  }
	
  /* fp = fopen(argv[1], "r"); */
  /* if (fp == NULL) { */
  /*   fprintf(stderr, "Error: File %s not found\n", argv[1]); */
  /*   exit (1); */
  /* } */
  /* ReadHMM(fp, &hmm); */
  /* fclose(fp); */

  fp = fopen(argv[1], "r");
  if (fp == NULL) {
    fprintf(stderr, "Error: File %s not found\n", argv[2]);
    exit (1);
  }
  ReadSequence(fp, &T, &O);
  fclose(fp);  
  
  ///TODO: generate a HMM model with the given parameters 
  hmm.N = atoi(argv[2]);
  hmm.M = atoi(argv[3]);
  hmm.A = dmatrix(1, hmm.N, 1, hmm.N);
  hmm.B = dmatrix(1, hmm.N, 1, hmm.M);
  hmm.pi = dvector(1, hmm.N);
  GenRandHMM(&hmm);  


  //printf("T\t N\t M\t gpu_time\t cpu_time\n");
  q = ivector(1,T);

  delta = dmatrix(1, T, 1, hmm.N);
  psi = imatrix(1, T, 1, hmm.N);

  //printf("------------------------------------\n");
  //printf("Viterbi using direct probabilities\n");
  
  double gpu_time = wallclock();
  ViterbiGPU(&hmm, T, O, delta, psi, q, &proba); 
  gpu_time = wallclock() - gpu_time;
  
  real gpu_res = log(proba);
  //fprintf(stdout, "Viterbi  MLE log prob = %E\n", log(proba));
  //fprintf(stdout, "Optimal state sequence:\n");
  //PrintSequence(stdout, T, q);

  free_ivector(q, 1, T);
  free_imatrix(psi, 1, T, 1, hmm.N);
  free_dmatrix(delta, 1, T, 1, hmm.N);

  q = ivector(1,T);
  delta = dmatrix(1, T, 1, hmm.N);
  psi = imatrix(1, T, 1, hmm.N);
  

  //printf("------------------------------------\n");
  //printf("Viterbi using direct probabilities\n");
  double cpu_time = wallclock();  
  Viterbi(&hmm, T, O, delta, psi, q, &proba); 
  cpu_time = wallclock() - cpu_time;
  //fprintf(stdout, "Viterbi  MLE log prob = %E\n", log(proba));
  //fprintf(stdout, "Optimal state sequence:\n");
  //PrintSequence(stdout, T, q);
  printf("%d\t %d\t %d\t %f\t %f\t ", T, hmm.N, hmm.M, gpu_time, cpu_time);
  real cpu_res = log(proba);
  if ( fabs(cpu_res -gpu_res) < 1e-7 )
    printf("passed\n");
  else
    printf("failed\n");

  /* printf("------------------------------------\n"); */
  /* printf("Viterbi using log probabilities\n"); */
  /* /\* note: ViterbiLog() returns back with log(A[i][j]) instead */
  /* ** of leaving the A matrix alone. If you need the original A, */
  /* ** you can make a copy of hmm by calling CopyHMM *\/  */

  /* ViterbiLog(&hmm, T, O, delta, psi, q, &logproba);  */

  /* fprintf(stdout, "Viterbi  MLE log prob = %E\n", logproba); */
  /* fprintf(stdout, "Optimal state sequence:\n"); */
  /* PrintSequence(stdout, T, q); */
  /* printf("------------------------------------\n"); */
  /* printf("The two log probabilites and optimal state sequences\n"); */
  /* printf("should identical (within numerical precision). \n"); */
	
  free_ivector(q, 1, T);
  free_ivector(O, 1, T);
  free_imatrix(psi, 1, T, 1, hmm.N);
  free_dmatrix(delta, 1, T, 1, hmm.N);
  FreeHMM(&hmm);
}

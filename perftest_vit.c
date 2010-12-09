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

//static char rcsid[] = "$Id: testvit.c,v 1.3 1998/02/23 07:39:07 kanungo Exp kanungo $";


void ViterbiMKL_Test_Header()
{  
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      printf("%d threads\t ", p);
    }             
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

  if (argc != 4) 
    {
      printf("Usage error \n");
      //printf("Usage: testvit <model.hmm> <obs.seq> \n");
      printf("Usage: testvit <sequence>  <num_states> <num_symbols> \n");
      exit (1);
    }

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

  
  static int first = 0;
  /* printf("------------------------------------\n"); */
  printf("T\t N\t M\t ");
  //printf("%d\t %d\t %d\t ", T, phmm->N, phmm->M);
  
  /* printf("Forward without scaling with CUDA \n"); */
  //double gpu_time = wallclock();
  ViterbiGPU_Test_Header();
  ViterbiMKL_Test_Header();
  printf("\n");

  printf("%d\t %d\t %d\t ", T, hmm.N, hmm.M);
  q = ivector(1,T);
  delta = dmatrix(1, T, 1, hmm.N);
  psi = imatrix(1, T, 1, hmm.N);               
  
  ViterbiGPU_Test_Driver(&hmm, T, O, delta, psi, q, &proba);
  ViterbiMKL_Test_Driver(&hmm, T, O, delta, psi, q, &proba);

  printf("\n");


  free_ivector(O, 1, T);
  dmatrix(delta, 1, T, 1, hmm.N);
  free_imatrix(psi, 1, T, 1, hmm.N);                 
  FreeHMM(&hmm);
}


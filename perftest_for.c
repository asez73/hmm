/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   4 May 1999 
**      File:   testfor.c
**      Purpose: driver for testing the Forward, ForwardWithScale code.
**      Organization: University of Maryland
**
**	$Id$
*/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <omp.h>
#include "nrutil.h"
#include "hmm.h"
//static char rcsid[] = "$Id: testvit.c,v 1.3 1998/02/23 07:39:07 kanungo Exp kanungo $";

//void ForwardMKL_Test_Header();
//void ForwardScaleMKL_Test_Header();



void ForwardMKL_Test_Header()
{  
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      //omp_set_num_threads(p);
      //cout<<deviceProp.name<<"\t";
      printf("%d threads\t ", p);
    }           
  //cout<<endl;
}


void ForwardScaleMKL_Test_Header()
{  
  int nproc = omp_get_max_threads();
  int p;
  for (p = 1; p <= nproc; p += 1) 
    {
      //omp_set_num_threads(p);
      //cout<<deviceProp.name<<"\t";
      printf("%d threads\t ", p);
    }           
  //cout<<endl;
}



int main (int argc, char **argv)
{
  int 	t, T; 
  HMM  	hmm;
  int	*O;	/* observation sequence O[1..T] */
  real **alpha;
  real *scale;
  real 	proba, logproba; 
  FILE	*fp;
  
  if (argc != 4) 
    {
      printf("Usage error \n");
      printf("Usage: testfor <obs.seq> N M\n");
      exit (1);
    }

  /* fp = fopen(argv[1], "r"); */
  /* if (fp == NULL) { */
  /*   fprintf(stderr, "Error: File %s not found\n", argv[1]); */
  /*   exit (1); */
  /* } */
  /* ReadHMM(fp, &hmm); */
  /* fclose(fp); */
	
  /* fp = fopen(argv[1], "r"); */
  /* if (fp == NULL) { */
  /*   fprintf(stderr, "Error: File %s not found\n", argv[2]); */
  /*   exit (1); */
  /* } */
  /* ReadSequence(fp, &T, &O); */
  /* fclose(fp); */
	
  fp = fopen(argv[1], "r");
  if (fp == NULL) 
    {
      fprintf(stderr, "Error: File %s not found\n", argv[2]);
      exit (1);
    }
  ReadSequence(fp, &T, &O);
  fclose(fp);  
  
  /* printf("------------------------------------\n"); */
  printf("T\t N\t M\t ");
  //printf("%d\t %d\t %d\t ", T, hmm.N, hmm.M);
  
  /* printf("Forward without scaling with CUDA \n"); */
  //double gpu_time = wallclock();
  ForwardGPU_Test_Header();
  ForwardMKL_Test_Header();
  ForwardScaleMKL_Test_Header();
  printf("\n");

  
  
  ///TODO: generate a HMM model with the given parameters 
  hmm.N = atoi(argv[2]);
  hmm.M = atoi(argv[3]);
  hmm.A = dmatrix(1, hmm.N, 1, hmm.N);
  hmm.B = dmatrix(1, hmm.N, 1, hmm.M);
  hmm.pi = dvector(1, hmm.N);
  GenRandHMM(&hmm);  
	
  //alpha = dmatrix(1, T, 1, hmm.N);
  //scale = dvector(1, T);
  alpha = matrix(1, T, 1, hmm.N);
	
       
  ForwardGPU_Test_Driver(&hmm, T, O, alpha, &proba); 
  //gpu_time = wallclock() - gpu_time;
  real gpu_res = log(proba);
  //fprintf(stderr, "log prob(O| model) = %E\n", log(proba));
  
  free_dmatrix(alpha, 1, T, 1, hmm.N);
  alpha = matrix(1, T, 1, hmm.N);

  /* printf("------------------------------------\n"); */
  /* printf("Forward without scaling with MKL\n"); */
  //double cpu_time = wallclock();
  //Forward(&hmm, T, O, alpha, &proba); 
  ForwardMKL_Test_Driver(&hmm, T, O, alpha, &proba); 
  //cpu_time = wallclock() - cpu_time; 
  real cpu_res = log(proba);
  //fprintf(stdout, "log prob(O| model) = %E\n", log(proba));
  

  /* printf("------------------------------------\n"); */
  /* printf("Forward with scaling using MKL\n"); */
  scale = vector(1, T);
  /* double cpu_scale_time = wallclock(); */
  ForwardScaleMKL_Test_Driver(&hmm, T, O, alpha, scale, &logproba);
  //ForwardWithScale(&hmm, T, O, alpha, scale, &logproba);
  /* cpu_scale_time = wallclock() - cpu_scale_time; */
  real cpu_scale_res = logproba; 


  printf("\n");
  //printf("%d\t %d\t %d\t %f\t %f\t %f\t", T, hmm.N, hmm.M, gpu_time, cpu_time, cpu_scale_time);  
  //printf("\n\t\t\t %f GFLOPS\t %f GFLOPS\t %f GFLOPS\t", (T*(hmm.N+1)*(hmm.N)*1e-6)/gpu_time, (T*(hmm.N+1)*(hmm.N)*1e-6)/cpu_time, (T*(hmm.N+1)*(hmm.N)*1e-6)/cpu_scale_time);  
  /* if ( (fabs(cpu_res-gpu_res) < 1e-4) && (fabs(gpu_res-cpu_scale_res) < 1e-4) ) */
  /*   printf("passed\n"); */
  /* else */
  /*   printf("failed\n gpu_res %f, cpu_res %f, cpu_scale_res %f\n", gpu_res, cpu_res, cpu_scale_res); */

  

  /* fprintf(stdout, "log prob(O| model) = %E\n", logproba); */
  /* printf("------------------------------------\n"); */
  /* printf("The two log probabilites should identical \n"); */
  /* printf("(within numerical precision). When observation\n"); */
  /* printf("sequence is very large, use scaling. \n"); */
	
  free_ivector(O, 1, T);
  free_dmatrix(alpha, 1, T, 1, hmm.N);
  free_dvector(scale, 1, T);
  FreeHMM(&hmm);
}


/*
**      File:   nrutil.c
**      Purpose: Memory allocation routines borrowed from the
**		book "Numerical Recipes" by Press, Flannery, Teukolsky,
**		and Vetterling. 
**              state sequence and probablity of observing a sequence
**              given the model.
**      Organization: University of Maryland
**
**      $Id: nrutil.c,v 1.2 1998/02/19 16:31:35 kanungo Exp kanungo $
*/

//#include <calloc.h> // calloc is now moved to stdlib.h
#include "hmm.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
static char rcsid[] = "$Id: nrutil.c,v 1.2 1998/02/19 16:31:35 kanungo Exp kanungo $";



void data_format(HMM *phmm, real** A, real** B, real** pi)
{
  int N = phmm->N;
  int M = phmm->M;  
  *A = (real*)malloc(N*N*sizeof(real));
  *B = (real*)malloc(N*M*sizeof(real));  
  *pi = (real*)malloc(N*sizeof(real));

  int i,j;
  for ( i=0; i<N; ++i )
    {
      (*pi)[i] = phmm->pi[i+1];
      for ( j=0; j<N; ++j)
  	(*A)[j*N + i] = phmm->A[i+1][j+1];
      for ( j=0; j<M; ++j )
  	(*B)[j*N + i] = phmm->B[i+1][j+1];
    }
  

  /* phmm->A = *A; */
  /* phmm->B = *B; */
  /* phmm->pi = *pi; */
}


void data_format_log(HMM *phmm, real** A, real** B, real** pi)
{
  int N = phmm->N;
  int M = phmm->M;  
  *A = (real*)malloc(N*N*sizeof(real));
  *B = (real*)malloc(N*M*sizeof(real));  
  *pi = (real*)malloc(N*sizeof(real));

  int i,j;
  for ( i=0; i<N; ++i )
    {
      (*pi)[i] = log(phmm->pi[i+1]);
      for ( j=0; j<N; ++j)
  	(*A)[j*N + i] = log(phmm->A[i+1][j+1]);
      for ( j=0; j<M; ++j )
  	(*B)[j*N + i] = log(phmm->B[i+1][j+1]);
    }
  

  /* phmm->A = *A; */
  /* phmm->B = *B; */
  /* phmm->pi = *pi; */
}




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



void nrerror(error_text)
char error_text[];
{
	void exit();

	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}



real* vector(nl,nh)
int nl,nh;
{
	real*v;

	v=(real*)calloc((unsigned) (nh-nl+1),sizeof(real));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl;
}

int *ivector(nl,nh)
int nl,nh;
{
	int *v;

	v=(int *)calloc((unsigned) (nh-nl+1),sizeof(int));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl;
}

real *dvector(nl,nh)
int nl,nh;
{
	real *v;

	v=(real *)calloc((unsigned) (nh-nl+1),sizeof(real));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl;
}



real** matrix(nrl,nrh,ncl,nch)
int nrl,nrh,ncl,nch;
{
	int i;
	real**m;

	m=(real**) calloc((unsigned) (nrh-nrl+1),sizeof(real*));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(real*) calloc((unsigned) (nch-ncl+1),sizeof(real));
		if (!m[i]) nrerror("allocation failure 2 in matrix()");
		m[i] -= ncl;
	}
	return m;
}

real** dmatrix(nrl,nrh,ncl,nch)
int nrl,nrh,ncl,nch;
{
	int i;
	real **m;

	m=(real **) calloc((unsigned) (nrh-nrl+1),sizeof(real*));
	if (!m) nrerror("allocation failure 1 in dmatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(real *) calloc((unsigned) (nch-ncl+1),sizeof(real));
		if (!m[i]) nrerror("allocation failure 2 in dmatrix()");
		m[i] -= ncl;
	}
	return m;
}

int **imatrix(nrl,nrh,ncl,nch)
int nrl,nrh,ncl,nch;
{
	int i,**m;

	m=(int **)calloc((unsigned) (nrh-nrl+1),sizeof(int*));
	if (!m) nrerror("allocation failure 1 in imatrix()");
	m -= nrl;

	for(i=nrl;i<=nrh;i++) {
		m[i]=(int *)calloc((unsigned) (nch-ncl+1),sizeof(int));
		if (!m[i]) nrerror("allocation failure 2 in imatrix()");
		m[i] -= ncl;
	}
	return m;
}



real** submatrix(a,oldrl,oldrh,oldcl,oldch,newrl,newcl)
real**a;
int oldrl,oldrh,oldcl,oldch,newrl,newcl;
{
	int i,j;
	real**m;

	m=(real**) calloc((unsigned) (oldrh-oldrl+1),sizeof(real*));
	if (!m) nrerror("allocation failure in submatrix()");
	m -= newrl;

	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+oldcl-newcl;

	return m;
}



void free_vector(v,nl,nh)
real*v;
int nl,nh;
{
	free((char*) (v+nl));
}

void free_ivector(v,nl,nh)
int *v,nl,nh;
{
	free((char*) (v+nl));
}

void free_dvector(v,nl,nh)
real *v;
int nl,nh;
{
	free((char*) (v+nl));
}



void free_matrix(m,nrl,nrh,ncl,nch)
real**m;
int nrl,nrh,ncl,nch;
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_dmatrix(m,nrl,nrh,ncl,nch)
real **m;
int nrl,nrh,ncl,nch;
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

void free_imatrix(m,nrl,nrh,ncl,nch)
int **m;
int nrl,nrh,ncl,nch;
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}



void free_submatrix(b,nrl,nrh,ncl,nch)
real **b;
int nrl,nrh,ncl,nch;
{
	free((char*) (b+nrl));
}



real **convert_matrix(a,nrl,nrh,ncl,nch)
real *a;
int nrl,nrh,ncl,nch;
{
	int i,j,nrow,ncol;
	real **m;

	nrow=nrh-nrl+1;
	ncol=nch-ncl+1;
	m = (real **) calloc((unsigned) (nrow),sizeof(real*));
	if (!m) nrerror("allocation failure in convert_matrix()");
	m -= nrl;
	for(i=0,j=nrl;i<=nrow-1;i++,j++) m[j]=a+ncol*i-ncl;
	return m;
}



void free_convert_matrix(b,nrl,nrh,ncl,nch)
real **b;
int nrl,nrh,ncl,nch;
{
	free((char*) (b+nrl));
}

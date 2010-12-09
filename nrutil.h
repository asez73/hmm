/*
**      File:   nrutil.h
**      Purpose: Memory allocation routines borrowed from the
**              book "Numerical Recipes" by Press, Flannery, Teukolsky,
**              and Vetterling.
**              state sequence and probablity of observing a sequence
**              given the model.
**      Organization: University of Maryland
**
**      $Id: nrutil.h,v 1.2 1998/02/19 16:32:42 kanungo Exp kanungo $
*/

#include "hmm.h"

void data_format(HMM *phmm, real** A, real** B, real** pi);
double wallclock(void);

real *vector();
real **matrix();
real **convert_matrix();

real *dvector();
real **dmatrix();

int *ivector();
int **imatrix();
real **submatrix();

void free_vector();
void free_dvector();
void free_ivector();
void free_matrix();
void free_dmatrix();
void free_imatrix();
void free_submatrix();
void free_convert_matrix();
void nrerror();

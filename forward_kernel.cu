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


//#define real float

/// this kernel deals with 
/**
 *\brief Forward O(N^2) inner loop
 *
 *\param Symbol the T-th observed symbol
 *\param delta_prev previous delta vector
 *\param delta_curr current delta vector to be updated
 *\param A the state transition matrix, in column major order
 *\param B the state emission matrix, in column major order
 *\param N number of states
 */
__global__ void ForwardKernel(int Symbol, real* alpha_prev, real* alpha_curr, real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  int pb_stride = blockDim.x * gridDim.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //int stride = num_threads / size;

  // each thread to be fixed at the tid-th column
  for ( int j=idx; j<N; j += pb_stride )
    { 
      real sumval = 0.0;
      //real maxvalidx = 1;
      for ( int i=0; i<N; ++i )
	{
	  sumval += alpha_prev[i] * A[j*N + i];
	}

      alpha_curr[j] = sumval * B[ (Symbol-1)*N + j ];
    }  
}



/// this kernel deals with 
/**
 *\brief Viterbi O(N^2) inner loop
 *
 * Compared with the basic version, this new kernel
 * makes each block compute one column
 *
 *\param Symbol the T-th observed symbol
 *\param delta_prev previous delta vector
 *\param delta_curr current delta vector to be updated
 *\param A the state transition matrix, in column major order
 *\param B the state emission matrix, in column major order
 *\param N number of states
 */
__global__ void ForwardKernelv1(int Symbol, real* alpha_prev, real* alpha_curr, real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  int pb_stride = gridDim.x;
  //int stride = num_threads / size;

  /// note that this kernel only takes 32 threads 
  const int num_threads = 32;
  __shared__ real maxvalist[num_threads];
  __shared__ int maxvalidx_list[num_threads];
    

  for ( int j=blockIdx.x; j<N; j += pb_stride )
    { 

      /// each thread to be fixed at the tid-th column
      real maxval = 0.0;
      real maxvalidx = 1;
      
      for ( int i=threadIdx.x; i<N; i += blockDim.x )
	{
	  real tmp = alpha_prev[i] * A[j*N + i];
	  if ( maxval < tmp )
	    {
	       maxval = tmp;
	       /// Waring: the index must be incremented
	       maxvalidx = i + 1;
	    }
	}
      maxvalist[threadIdx.x] = maxval;
      maxvalidx_list[threadIdx.x] = maxvalidx;

      /// block level barrier
      __syncthreads();
      
      if ( threadIdx.x == 0 )
	{
	  maxval = maxvalist[0];
	  maxvalidx = maxvalidx_list[0];
	  for ( int t=1; t<num_threads; ++t )
	    {
	      if ( maxval < maxvalist[t] )
		{
		  maxval = maxvalist[t];
		  maxvalidx = maxvalidx_list[t];
		}
	    }
	  
	  alpha_curr[j] = maxval * B[ (Symbol-1)*N + j ];
	}
    }  
}


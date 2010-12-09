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
 *\brief Viterbi O(N^2) inner loop
 *
 *\param Symbol the T-th observed symbol
 *\param delta_prev previous delta vector
 *\param delta_curr current delta vector to be updated
 *\param A the state transition matrix, in column major order
 *\param B the state emission matrix, in column major order
 *\param N number of states
 */
__global__ void ViterbiKernel(int Symbol, real* delta_prev, real* delta_curr, int* psi_curr, real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  int pb_stride = blockDim.x * gridDim.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //int stride = num_threads / size;

  // each thread to be fixed at the tid-th column
  for ( int j=idx; j<N; j += pb_stride )
    { 
      real maxval = 0.0;
      real maxvalidx = 1;
      for ( int i=0; i<N; ++i )
	{
	  real tmp = delta_prev[i] * A[j*N + i];
	  if ( maxval < tmp )
	    {
	      maxval = tmp;
	      /// Waring: the index must be incremented
	      maxvalidx = i + 1;
	    }
	}
      delta_curr[j] = maxval * B[ (Symbol-1)*N + j ];
      psi_curr[j] = maxvalidx;      
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
__global__ void ViterbiKernelv1(int Symbol, real* delta_prev, real* delta_curr, int* psi_curr, real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  //int pb_stride = gridDim.x;
  //int stride = num_threads / size;

  /// note that this kernel only takes 32 threads 
  const int num_threads = 32;
  __shared__ real maxvalist[num_threads];
  __shared__ int maxvalidx_list[num_threads];
    
  for ( int j=blockIdx.x; j<N; j += gridDim.x )
    {
      /// each thread to be fixed at the tid-th column
      register real maxval = 0.0;
      register real maxvalidx = 1;
  
      for ( int i=threadIdx.x; i<N; i += blockDim.x )
  	{
  	  real tmp = delta_prev[i] * A[j*N + i];
  	  if ( maxval < tmp )
  	    {
	      maxval = tmp;
	      /// Waring: the index must be incremented
	      maxvalidx = i + 1;
  	    }
  	}
      
      maxvalist[threadIdx.x] = maxval;
      maxvalidx_list[threadIdx.x] = maxvalidx;
      __threadfence_block();
      //__syncthreads();
      
      /// we should consider using a better method
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
	  
      	  delta_curr[j] = maxval * B[ (Symbol-1)*N + j ];
      	  psi_curr[j] = maxvalidx;
      	}
      __syncthreads();
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
__global__ void ViterbiLogKernelv1(int Symbol, real* delta_prev, real* delta_curr, int* psi_curr, real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  //int pb_stride = gridDim.x;
  //int stride = num_threads / size;

  /// note that this kernel only takes 32 threads 
  const int num_threads = 32;
  __shared__ real maxvalist[num_threads];
  __shared__ int maxvalidx_list[num_threads];
    
  for ( int j=blockIdx.x; j<N; j += gridDim.x )
    {
      /// each thread to be fixed at the tid-th column
      register real maxval = 0.0;
      register real maxvalidx = 1;
  
      for ( int i=threadIdx.x; i<N; i += blockDim.x )
  	{
  	  real tmp = delta_prev[i] + log(A[j*N + i]);
  	  if ( maxval < tmp )
  	    {
	      maxval = tmp;
	      /// Waring: the index must be incremented
	      maxvalidx = i + 1;
  	    }
  	}
      
      maxvalist[threadIdx.x] = maxval;
      maxvalidx_list[threadIdx.x] = maxvalidx;
      __threadfence_block();
      //__syncthreads();
      
      /// we should consider using a better method
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
	  
      	  delta_curr[j] = maxval + log(B[ (Symbol-1)*N + j ]);
      	  psi_curr[j] = maxvalidx;
      	}
      __syncthreads();
    }  
}




__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;


__device__ void syncAllThreads()
{
  __syncthreads();
  if (threadIdx.x == 0)
    atomicInc(&count, gridDim.x-1);
  volatile unsigned int* counter = &count;
    do
      {
      } while (*counter > 0);
}

/**
 *\brief Viterbi O(TN^2) main loop
 *
 * Compared with the basic version, this new kernel
 * makes each block compute one column
 *
 *\param query the query observation sequence
 *\param delta_prev previous delta vector
 *\param delta_curr current delta vector to be updated
 *\param A the state transition matrix, in column major order
 *\param B the state emission matrix, in column major order
 *\param N number of states
 */
__global__ void ViterbiKernelv2(int* query, int T,
				real* delta, int* psi, 
				real* A, real *B, size_t N)
{
  /// compute the distribution of threads
  /// using 1D layout  

  //int pb_stride = gridDim.x;
  //int stride = num_threads / size;

  /// note that this kernel only takes 64 threads 
  const int num_threads = 32;
  __shared__ real maxvalist[num_threads];
  __shared__ int maxvalidx_list[num_threads];

  
  for ( int t=1; t<T; ++t )
    { 
      for ( int j = blockIdx.x; j<N; j += gridDim.x )
	{
	  /// each thread to be fixed at the tid-th column
	  register real maxval = 0.0;
	  register real maxvalidx = 1;

	  for ( int i=threadIdx.x; i<N; i += blockDim.x )
	    {
	      real tmp = delta[(t-1)*N + i] * A[j*N + i];
	      if ( maxval < tmp )
		{
		  maxval = tmp;
		  /// Waring: the index must be incremented
		  maxvalidx = i + 1;
		}
	    }
	  
	  maxvalist[threadIdx.x] = maxval;
	  maxvalidx_list[threadIdx.x] = maxvalidx;
	  __threadfence_block();
	  __syncthreads();
	  
      
	  /// update the last one here
	  if ( threadIdx.x == 0 )
	    {
	      maxval = maxvalist[0];
	      maxvalidx = maxvalidx_list[0];
	      for ( int btid=1; btid<num_threads; ++btid )
		{
		  if ( maxval < maxvalist[btid] )
		    {
		      maxval = maxvalist[btid];
		      maxvalidx = maxvalidx_list[btid];
		    }
		}
	  
	      delta[t*N + j] = maxval * B[(query[t] - 1)*N + j];
	      psi[t*N + j] = maxvalidx;      

	      __threadfence();
	    }
	  __syncthreads();
	}

      syncAllThreads();
      
    }  
}


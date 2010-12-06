#
#
# Make file for compiling HMM code in this directory.
# Author: Tapas Kanungo
# Date: 23 February 1998
# $Id: Makefile,v 1.3 1998/02/23 08:12:35 kanungo Exp kanungo $
# 
#

# single precision just doesn't work here...
CFLAGS= -g -D__GPU -D__DOUBLE 
INCS=
# use the following line to "Purify" the code
#CC=purify gcc

LN_FLAGS = -L/opt/intel/composerxe-2011.0.084/mkl/lib/intel64

CC=gcc
SRCS=baum.c viterbi.c forward.c backward.c hmmutils.c sequence.c \
	genseq.c nrutil.c testvit.c esthmm.c hmmrand.c testfor.c \
	viterbi_gpu.cu viterbi_kernel.cu forward_mkl.c \
	genrandhmm.c forward_gpu.cu forward_kernel.cu viterbi_mkl.c \
	perftest_vit.c perftest_for.c

EXE = genseq testvit testfor esthmm perftest_vit perftest_for

all : $(EXE)

# libhmm.so: baum.o viterbi.o forward.o backward.o nrutil.o hmmutils.o 
# 	$(CC) 

# viterbi_kernel.o: viterbi_kernel.cu 
# 	nvcc -c $^ $(CUDA_MAKEFILE_I) -arch=sm_13 -Xcompiler $(CFLAGS)

viterbi_gpu.o: viterbi_gpu.cu viterbi_kernel.cu
	nvcc -c $^ $(CUDA_MAKEFILE_I) -arch=sm_13 -Xcompiler $(CFLAGS)

forward_gpu.o: forward_gpu.cu forward_kernel.cu
	nvcc -c $^ $(CUDA_MAKEFILE_I) -arch=sm_13 -Xcompiler $(CFLAGS)

genseq: genseq.o sequence.o nrutil.o hmmutils.o  hmmrand.o
	 $(CC) -o $@ $^ -lm

# testvit: testvit.o viterbi.o nrutil.o hmmutils.o sequence.o hmmrand.o genrandhmm.o
# 	 $(CC) -o $@ $^ -lm

testvit: testvit.o viterbi_mkl.o viterbi_gpu.o nrutil.o hmmutils.o sequence.o hmmrand.o genrandhmm.o
	 $(CC) -o $@ $^ $(LN_FLAGS) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm $(CUDA_MAKEFILE_L) -lcuda -lcudart -fopenmp

testfor: testfor.o forward_gpu.o forward_mkl.o nrutil.o hmmutils.o sequence.o hmmrand.o genrandhmm.o
	 $(CC) -o $@ $^ $(LN_FLAGS) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm $(CUDA_MAKEFILE_L) -lcuda -lcudart -fopenmp


perftest_vit: perftest_vit.o viterbi_mkl.o viterbi_gpu.o nrutil.o hmmutils.o sequence.o hmmrand.o genrandhmm.o
	 $(CC) -o $@ $^ $(LN_FLAGS) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm $(CUDA_MAKEFILE_L) -lcuda -lcudart -fopenmp


perftest_for: perftest_for.o forward_gpu.o forward_mkl.o nrutil.o hmmutils.o sequence.o hmmrand.o genrandhmm.o
	 $(CC) -o $@ $^ $(LN_FLAGS) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm $(CUDA_MAKEFILE_L) -lcuda -lcudart -fopenmp


# testfor: testfor.o forward.o nrutil.o hmmutils.o sequence.o hmmrand.o
# 	 $(CC) -o $@ $^ -lm 

esthmm: esthmm.o baum.o nrutil.o hmmutils.o sequence.o \
		forward.o backward.o hmmrand.o
	 $(CC) -o $@ $^ -lm

clean:
	rm *.o $(EXE)
# DO NOT DELETE THIS LINE -- make depend depends on it.


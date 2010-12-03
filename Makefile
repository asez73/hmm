#
#
# Make file for compiling HMM code in this directory.
# Author: Tapas Kanungo
# Date: 23 February 1998
# $Id: Makefile,v 1.3 1998/02/23 08:12:35 kanungo Exp kanungo $
# 
#
CFLAGS= -g -D__GPU
INCS=
# use the following line to "Purify" the code
#CC=purify gcc
CC=gcc
SRCS=baum.c viterbi.c forward.c backward.c hmmutils.c sequence.c \
	genseq.c nrutil.c testvit.c esthmm.c hmmrand.c testfor.c \
	viterbi_gpu.c
EXE = genseq testvit testfor esthmm testvit_gpu

all :	genseq testvit testfor esthmm testvit_gpu

viterbi_gpu.o: viterbi_gpu.c
	$(CC) -c $^ $(CUDA_MAKEFILE_I) -D__GPU 

genseq: genseq.o sequence.o nrutil.o hmmutils.o  hmmrand.o
	 $(CC) -o $@ $^ -lm

testvit: testvit.o viterbi.o nrutil.o hmmutils.o sequence.o hmmrand.o
	 $(CC) -o $@ $^ -lm

testvit_gpu: testvit.o viterbi_gpu.o nrutil.o hmmutils.o sequence.o hmmrand.o
	 $(CC) -o $@ $^ -lm $(CUDA_MAKEFILE_L) -lcuda -lcudart -fopenmp

testfor: testfor.o forward.o nrutil.o hmmutils.o sequence.o hmmrand.o
	 $(CC) -o $@ $^ -lm 

esthmm: esthmm.o baum.o nrutil.o hmmutils.o sequence.o \
		forward.o backward.o hmmrand.o
	 $(CC) -o $@ $^ -lm

clean:
	rm *.o $(EXE)
# DO NOT DELETE THIS LINE -- make depend depends on it.


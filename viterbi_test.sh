#!/usr/bin/bash

# check the machine name
uname -a 

# get the data
for N in $(seq 512 512 8192); do ./perftest_vit t2.50.seq $N 65; done | tee T50_Forward_opengpu.res
for N in $(seq 512 512 8192); do ./perftest_for t2.50.seq $N 65; done | tee T50_Forward_opengpu.res
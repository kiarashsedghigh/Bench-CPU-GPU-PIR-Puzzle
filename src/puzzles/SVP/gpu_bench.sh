#!/bin/bash

nvcc -w -O3 puzzle_gen_gpu.cu -o svp_puzzle_gpu
./svp_puzzle_gpu

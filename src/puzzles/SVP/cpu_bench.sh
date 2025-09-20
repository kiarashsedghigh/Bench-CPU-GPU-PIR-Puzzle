#!/bin/bash

g++ -w puzzle_gen_cpu.cpp -O3 -o svp_puzzle_cpu -lntl
./svp_puzzle_cpu

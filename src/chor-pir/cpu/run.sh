#!/bin/bash

# Define arrays of q and r values
q_values=(1 2 3 4 5 6 7 8 9 10)
r_values=(11 12 13 14 15 16 17)

for q in "${q_values[@]}"; do
    for r in "${r_values[@]}"; do
        echo "Running: ./main q=$q r=$r"
        ./main q=$q r=$r
    done
done

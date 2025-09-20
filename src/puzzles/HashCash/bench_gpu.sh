#!/bin/bash

nvcc -w hashcashtree_gen.cu -o hashcashtree_gen_gpu -O3

# Configuration
ITERATIONS=100
PROGRAM="./hashcashtree_gen_gpu "  # Replace with actual executable path

# Initialize sum
sum=0

echo "Running $PROGRAM $ITERATIONS times..."

for ((i = 1; i <= ITERATIONS; i++)); do
    # Run the program and capture its output
    output=$($PROGRAM)

    # Extract the time value using regex (floating point number)
    time=$(echo "$output" | grep -oP 'Kernel execution time:\s*\K[0-9.]+')

    if [[ -z "$time" ]]; then
        echo "Failed to parse time on iteration $i"
        continue
    fi

    echo "Run $i: $time ms"

    # Convert to integer microseconds for summing: time * 1000
    # Use bc for floating-point arithmetic
    micros=$(echo "$time * 1000" | bc)
    sum=$(echo "$sum + $micros" | bc)
done

# Compute average in microseconds, then convert to milliseconds
average_micro=$(echo "$sum / $ITERATIONS" | bc -l)
average_milli=$(echo "scale=3; $average_micro / 1000" | bc)

echo "Average time: $average_milli ms"

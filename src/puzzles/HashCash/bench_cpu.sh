#!/bin/bash

g++ -w -O3 hashcashtree_gen.cpp -lssl -lcrypto -o hashcashtree_gen_cpp

# Configurable: number of iterations and your executable
ITERATIONS=1000
PROGRAM="./hashcashtree_gen_cpp"  # Replace with your actual binary path

# Initialize sum
sum=0

echo "Running $PROGRAM $ITERATIONS times..."

for ((i = 1; i <= ITERATIONS; i++)); do
    # Run the program and capture the output
    output=$($PROGRAM)

    # Extract last number before "micro s"
    time=$(echo "$output" | grep -oP 'in \K[0-9]+(?=micro s)')
    if [[ -z "$time" ]]; then
        echo "Failed to parse time on iteration $i"
        continue
    fi

    echo "Run $i: $time microseconds"
    sum=$((sum + time))
done

# Compute average
average=$((sum / ITERATIONS))
echo "Average time: $average microseconds"
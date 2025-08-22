nvcc gpu_mm_mult.cu -o gpu_main -O3
mkdir -p gpu-times  # Ensure output directory exists

for db_exp in {10,12,14,16}; do
    db_size=$((2**db_exp))
    echo $db_size
    for query_exp in {0,7,10}; do
        query_count=$((2**query_exp))
        filename="gpu-times/time${db_exp}-${query_count}.txt"

        echo "2^$db_exp DB, $query_count Queries"
        > "$filename"

        for i in {1..10}; do
            echo "Run #$i"
            ./gpu_main "$query_count" "$db_size" 2048 2 3 2 >> "$filename" 2>&1
        done
    done
done

g++ -w  cpu_mm_mult.cpp -o cpu_main -mavx2 -O3
mkdir -p cpu-times  # Ensure output directory exists

for db_exp in {18,}; do
    db_size=$((2**db_exp))
    echo $db_size
    for query_exp in {0,}; do
        query_count=$((2**query_exp))
        filename="cpu-times/time${db_exp}-${query_count}.txt"

        echo "2^$db_exp DB, $query_count Queries"
        > "$filename"

        for i in {1,}; do
            echo "Run #$i"
            ./cpu_main "$query_count" "$db_size" 2048 2 3 2 >> "$filename" 2>&1
        done
    done
done

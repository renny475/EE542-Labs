#!/usr/bin/env bash
set -euo pipefail

sizes=(128 256 384 512 640 768 896 1024 1280 1536 1792 2048)
runs=3
output="cublas_results.csv"

echo "N,run,time_ms" > "$output"

for N in "${sizes[@]}"; do
    for run in $(seq 1 "$runs"); do
        output_line=$(./matrixcublas "$N" | grep "cuBLAS SGEMM average time")

        echo "Trial $run/$runs for N=$N"
        echo "$output_line"

        time_ms=$(echo "$output_line" | awk '{print $(NF-1)}')
        echo "$N,$run,$time_ms" >> "$output"
    done
done

echo "Saved cuBLAS results to $output"

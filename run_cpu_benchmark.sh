#!/usr/bin/env bash
set -euo pipefail

sizes=(128 256 384 512 640 768 896 1024 1280 1536 1792 2048)
runs=3
output="cpu_results.csv"

echo "N,run,time_seconds" > "$output"

for N in "${sizes[@]}"; do
    for run in $(seq 1 "$runs"); do
        echo "Trial $run/$runs for N=$N"

        result=$(./matrixcpu "$N")
        echo "$result"

        time_seconds=$(echo "$result" | awk '{print $(NF-1)}')
        echo "$N,$run,$time_seconds" >> "$output"
    done
done

echo "Results saved to $output"

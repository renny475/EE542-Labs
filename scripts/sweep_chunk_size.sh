#!/usr/bin/env bash
# Runs the receiver across a list of chunk sizes, one after another, and
# logs each round's throughput to a CSV. The matching client command is
# printed each round for you to run by hand on the other VM (both sides
# still have to be started fresh together, same as any other run - see
# the Lab2 Protocol Log for why).
#
# Run this on the SERVER VM. Set the network condition (tc config) for
# whichever case you're sweeping *before* starting this script - it does
# not touch tc itself.
#
# Usage:
#   ./sweep_chunk_size.sh <port> <output_prefix> <chunk_size1> [chunk_size2 ...]
#
# Example (MTU 1500 candidates):
#   ./sweep_chunk_size.sh 9999 swept_case1 1024 1200 1400 1456
#
# Example (MTU 9001 candidates):
#   ./sweep_chunk_size.sh 9999 swept_case1_jumbo 2048 4500 8900

set -u

if [ $# -lt 3 ]; then
    echo "usage: $0 <port> <output_prefix> <chunk_size1> [chunk_size2 ...]" >&2
    exit 1
fi

PORT=$1
PREFIX=$2
shift 2
SIZES=("$@")

RESULTS="sweep_results_$(date +%Y%m%d_%H%M%S).csv"
echo "chunk_size,elapsed_s,throughput_mbit,bytes_received,gave_up" > "$RESULTS"

echo "Results will be logged to: $RESULTS"

for SIZE in "${SIZES[@]}"; do
    OUTFILE="${PREFIX}_${SIZE}.bin"
    LOG=$(mktemp)

    echo
    echo "=================================================================="
    echo " chunk_size=$SIZE -- starting server, output=$OUTFILE"
    echo
    echo " On the CLIENT VM, once you see \"Waiting for file data...\" below,"
    echo " run (server first, this already is server):"
    echo
    echo "   ./client <server_ip> $PORT testfile.bin $SIZE"
    echo "=================================================================="
    echo

    ./server "$PORT" "$OUTFILE" "$SIZE" 2>&1 | tee "$LOG"

    ELAPSED=$(grep "Elapsed time:" "$LOG" | grep -oE '[0-9.]+' | head -1)
    THROUGHPUT=$(grep "Receive throughput:" "$LOG" | grep -oE '[0-9.]+' | head -1)
    BYTES=$(grep "Bytes received:" "$LOG" | grep -oE '[0-9]+' | head -1)
    GAVEUP="no"
    grep -q "WARNING: gave up" "$LOG" && GAVEUP="yes"

    echo "$SIZE,${ELAPSED:-NA},${THROUGHPUT:-NA},${BYTES:-NA},$GAVEUP" >> "$RESULTS"
    rm -f "$LOG"

    echo
    echo ">>> chunk_size=$SIZE done. Running total in $RESULTS:"
    cat "$RESULTS"
    echo
    echo ">>> Press Enter once the client side is idle and ready for the next size..."
    read -r _
done

echo
echo "All rounds complete."
echo "$RESULTS:"
cat "$RESULTS"
echo
echo "Pick the best throughput here, then run ONE full round on that"
echo "chunk_size with MD5 verification on both sides before recording it"
echo "as your result for this case - the sweep itself only ranks candidates."

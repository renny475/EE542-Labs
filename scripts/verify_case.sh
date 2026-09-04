#!/usr/bin/env bash
# Measures RTT and loss to a peer via ping, then reports which of the
# lab's three mandatory cases (if any) the measurement matches. Run
# this on EITHER the client or server VM, pointed at the other side,
# right before starting a file-transfer test - not a substitute for
# running it in both directions per the lab spec, just a quick check
# to catch "wrong tc config still active" before you waste a run.
#
# Usage: ./verify_case.sh <peer_ip> [ping_count]

set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <peer_ip> [ping_count]" >&2
    exit 1
fi

PEER=$1
COUNT=${2:-200}

echo "Pinging $PEER ($COUNT packets, 0.2s interval - matches the lab's spec)..."
echo

PING_OUT=$(ping -i 0.2 -c "$COUNT" "$PEER" 2>&1)
echo "$PING_OUT"
echo

# Linux ping summary lines look like:
#   200 packets transmitted, 158 received, 21% packet loss, time 40450ms
#   200 packets transmitted, 193 received, 3.5% packet loss, time 40250ms
#   rtt min/avg/max/mdev = 200.906/201.086/202.648/0.319 ms
LOSS=$(echo "$PING_OUT" | grep -oE '[0-9]+(\.[0-9]+)?% packet loss' | grep -oE '^[0-9]+(\.[0-9]+)?')
RTT_AVG=$(echo "$PING_OUT" | grep 'rtt min/avg/max' | sed -E 's#.*= ([0-9.]+)/([0-9.]+)/.*#\2#')

if [ -z "$LOSS" ] || [ -z "$RTT_AVG" ]; then
    echo "Could not parse ping output (unexpected format) - read it manually above."
    exit 1
fi

echo "=================================================================="
echo " Measured: RTT avg ~= ${RTT_AVG}ms, loss ~= ${LOSS}%"
echo "=================================================================="
echo

# Case profiles from the lab spec, with the +-20% loss tolerance it allows.
awk -v rtt="$RTT_AVG" -v loss="$LOSS" '
BEGIN {
    # case, expect_rtt_lo, expect_rtt_hi, expect_loss_lo, expect_loss_hi
    n = 3
    name[1]="Case 1"; rtt_lo[1]=8;   rtt_hi[1]=25;  loss_lo[1]=0.8; loss_hi[1]=1.2
    name[2]="Case 2"; rtt_lo[2]=170; rtt_hi[2]=260; loss_lo[2]=16;  loss_hi[2]=24
    name[3]="Case 3"; rtt_lo[3]=170; rtt_hi[3]=260; loss_lo[3]=0;   loss_hi[3]=2

    matched = 0
    for (i = 1; i <= n; i++) {
        rtt_ok = (rtt+0 >= rtt_lo[i] && rtt+0 <= rtt_hi[i])
        loss_ok = (loss+0 >= loss_lo[i] && loss+0 <= loss_hi[i])
        if (rtt_ok && loss_ok) {
            print "MATCH: this looks like " name[i] " (RTT " rtt_lo[i] "-" rtt_hi[i] "ms, loss " loss_lo[i] "-" loss_hi[i] "%)."
            matched = 1
        }
    }
    if (!matched) {
        print "NO MATCH: measured RTT/loss does not fall within any of the three cases tolerance bands."
        print "Double-check the router tc config (tc qdisc show) before running your transfer test."
    }
}'

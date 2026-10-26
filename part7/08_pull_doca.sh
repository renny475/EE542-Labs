#!/usr/bin/env bash
# node0 only — start this right after reboot #1 (02_post_reboot.sh), it takes 10-20 min (22 GB).
# Runs in background; check with:  tail -f ~/logs/node0/docker_pull.log
IMG=nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host
LOG=$HOME/logs/$(hostname -s); mkdir -p "$LOG"
sudo docker info | grep -i 'docker root dir'          # must say /mydata/docker
nohup sudo docker pull "$IMG" > "$LOG/docker_pull.log" 2>&1 &
echo "pulling in background (pid $!) → $LOG/docker_pull.log"

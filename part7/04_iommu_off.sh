#!/usr/bin/env bash
# Part 7 / Step 7.6 — run on BOTH nodes. Turns the IOMMU off and reboots.
# After reboot: run 02_post_reboot.sh again, then `ibv_devices` (device name may change!).
set -euo pipefail
LOG=$HOME/logs/$(hostname -s); mkdir -p "$LOG"
exec > >(tee -a "$LOG/7.6_iommu.log") 2>&1
echo "=== [$(date)] 7.6 IOMMU off on $(hostname) ==="
grep -q 'amd_iommu=off' /etc/default/grub || \
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&amd_iommu=off iommu=off /' /etc/default/grub
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
sudo update-grub
echo "rebooting in 5s…"; sleep 5
sudo reboot

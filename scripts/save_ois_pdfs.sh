#!/usr/bin/env bash
# Save the OIS pages listed in data/sources.yaml as PDFs using the Chrome
# installed on this Mac (headless print-to-PDF). Run it yourself from the
# repo root:  bash scripts/save_ois_pdfs.sh
# It is the same as opening each page in Chrome and pressing Cmd+P > Save as
# PDF. Chrome identifies itself as HeadlessChrome; if OIS refuses that, the
# script reports the page and you save it by hand.
set -u
cd "$(dirname "$0")/.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT="data/ois"
mkdir -p "$OUT"

URLS=(
  https://ois.usc.edu/employment/f1-student-employment/
  https://ois.usc.edu/employment/on-campus-employment/
  https://ois.usc.edu/employment/curricular-practical-training-cpt/
  https://ois.usc.edu/employment/optional-practical-training-opt/
  https://ois.usc.edu/employment/pre-completion-opt/
  https://ois.usc.edu/employment/post-completion-opt/
  https://ois.usc.edu/employment/stem-opt-extension/
  https://ois.usc.edu/employment/stem-opt-extension-prior-degree/
  https://ois.usc.edu/employment/cap-gap-extension/
  https://ois.usc.edu/employment/additional-forms-of-employment/
  https://ois.usc.edu/employment/social-security-number/
  https://ois.usc.edu/maintaining-status/guidelines-maintaining-status/
  https://ois.usc.edu/maintaining-status/change-of-status-f1/
  https://ois.usc.edu/maintaining-status/extending-program-students/
  https://ois.usc.edu/maintaining-status/progressive-degree-program/
  https://ois.usc.edu/maintaining-status/attending-another-institution/
  https://ois.usc.edu/maintaining-status/reduced-course-load/
  https://ois.usc.edu/maintaining-status/leave-of-absence/
  https://ois.usc.edu/maintaining-status/program-completion-students/
  https://ois.usc.edu/travel/entering-united-states/
  https://ois.usc.edu/travel/renewing-passport-f1-j1-visa/
  https://ois.usc.edu/travel/requirements-for-reentry/
  https://ois.usc.edu/travel/automatic-revalidation/
  https://ois.usc.edu/getting-started/student-visa-types/
  https://ois.usc.edu/getting-started/transferring-to-usc/
  https://ois.usc.edu/getting-started/f2-dependents/
  https://ois.usc.edu/duration-status-rule-change/
  https://ois.usc.edu/dhs-final-rule-postponed/
  https://ois.usc.edu/update-regarding-proposed-opt-fees/
)

ok=0; bad=()
for url in "${URLS[@]}"; do
  slug="${url%/}"; slug="${slug##*/}"
  pdf="$OUT/$slug.pdf"
  if [ -s "$pdf" ]; then echo "skip   $slug.pdf (exists)"; ok=$((ok+1)); continue; fi
  "$CHROME" --headless=new --disable-gpu --no-first-run --no-default-browser-check \
    --user-data-dir=/tmp/ois-chrome-profile --virtual-time-budget=10000 \
    --no-pdf-header-footer --print-to-pdf="$pdf" "$url" >/dev/null 2>&1
  # a refused page prints as a tiny PDF; anything under 20 KB is suspect
  if [ -s "$pdf" ] && [ "$(stat -f%z "$pdf")" -gt 20000 ]; then
    echo "saved  $slug.pdf ($(stat -f%z "$pdf") bytes)"; ok=$((ok+1))
  else
    echo "FAILED $slug.pdf  -> open $url in Chrome and save it by hand"; rm -f "$pdf"; bad+=("$url")
  fi
  sleep 2
done
echo
echo "$ok of ${#URLS[@]} saved."
[ ${#bad[@]} -eq 0 ] || { echo "Save these manually:"; printf '  %s\n' "${bad[@]}"; exit 1; }

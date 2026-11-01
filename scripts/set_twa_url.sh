#!/usr/bin/env bash
# Point the Android TWA at the deployed web app and (optionally) record the signing
# fingerprint in assetlinks.json. Run from the repo root (groupchat_app_src/).
#
#   bash scripts/set_twa_url.sh https://my-tunnel.example.com [SHA256:AA:BB:...]
set -euo pipefail
url="${1:?usage: set_twa_url.sh https://YOUR_DOMAIN [SHA256 fingerprint]}"
fingerprint="${2:-}"
url="${url%/}"
case "$url" in https://*) ;; *) echo "the TWA needs an https URL, got: $url" >&2; exit 1 ;; esac

strings=twa_android_src/app/src/main/res/values/strings.xml
cat > "$strings" <<XML
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <string name="app_name">USC OIS Assistant</string>
    <!-- The deployed https URL of the web app. Set both values with: bash scripts/set_twa_url.sh https://YOUR_DOMAIN -->
    <string name="defaultUrl">${url}/</string>
    <string name="asset_statements" translatable="false">[{\"relation\": [\"delegate_permission/common.handle_all_urls\"], \"target\": {\"namespace\": \"web\", \"site\": \"${url}\"}}]</string>
</resources>
XML
echo "defaultUrl -> ${url}/"

if [ -n "$fingerprint" ]; then
  fingerprint="${fingerprint#SHA256:}"
  python3 - "$fingerprint" <<'PY'
import json, sys
path = "frontend/.well-known/assetlinks.json"
data = json.load(open(path))
data[0]["target"]["sha256_cert_fingerprints"] = [sys.argv[1]]
json.dump(data, open(path, "w"), indent=2)
open(path, "a").write("\n")
print("assetlinks.json fingerprint ->", sys.argv[1])
PY
fi

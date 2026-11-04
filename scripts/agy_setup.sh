#!/usr/bin/env bash
# Install the Antigravity CLI (`agy`) and restore its OAuth token, so the council can call Gemini.
# Same procedure as dlwjdgh95-byte/Miner blog/scripts/agy_setup.sh.
# Contract: never blocks the caller (always exit 0); on failure the council runs without Gemini.
set -uo pipefail
BIN_DIR="${AGY_BIN_DIR:-$HOME/.agy-bin}"   # under $HOME, not /tmp (anyone can write /tmp)
TOKEN_DIR="$HOME/.gemini/antigravity-cli"

if [[ -z "${AGY_OAUTH_TOKEN_B64:-}" ]]; then
    echo "agy: AGY_OAUTH_TOKEN_B64 not set"; exit 0
fi
if [[ ! -x "$BIN_DIR/agy" ]]; then
    mkdir -p "$BIN_DIR"
    log="$BIN_DIR/install.log"
    if ! { curl -fsSL https://antigravity.google/cli/install.sh -o "$BIN_DIR/install.sh" 2>"$log" \
           && bash "$BIN_DIR/install.sh" --dir "$BIN_DIR" >>"$log" 2>&1; } \
       || [[ ! -x "$BIN_DIR/agy" ]]; then
        echo "agy: install failed: $(grep -v '^\s*$' "$log" 2>/dev/null | tail -1 | cut -c1-160)"; exit 0
    fi
fi
umask 077
mkdir -p "$TOKEN_DIR"
if ! printf '%s' "$AGY_OAUTH_TOKEN_B64" | base64 -d > "$TOKEN_DIR/antigravity-oauth-token" 2>/dev/null \
   || [[ ! -s "$TOKEN_DIR/antigravity-oauth-token" ]]; then
    rm -f "$TOKEN_DIR/antigravity-oauth-token"   # an empty file would make the next run skip restoring
    echo "agy: token restore failed"; exit 0
fi
chmod 600 "$TOKEN_DIR/antigravity-oauth-token"
echo "agy: ready ($("$BIN_DIR/agy" --version 2>/dev/null | tail -1))"
exit 0

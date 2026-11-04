#!/usr/bin/env bash
# Ask one Gemini model a question through agy, for the in-session idea-council skill.
#   scripts/agy_ask.sh <model> <prompt-file>
# Runs in an empty temp directory with --sandbox, passing agy only HOME, PATH, locale and the
# proxy/CA variables (no other secrets). Prints the answer; exits non-zero on failure.
set -uo pipefail
model="${1:?model}"; prompt_file="${2:?prompt file}"
here="$(cd "$(dirname "$0")" && pwd)"
agy="${AGY_BIN:-$HOME/.agy-bin/agy}"
if [[ ! -x "$agy" || ! -s "$HOME/.gemini/antigravity-cli/antigravity-oauth-token" ]]; then
    bash "$here/agy_setup.sh" >&2
fi
[[ -x "$agy" ]] || { echo "agy not available" >&2; exit 2; }
prompt="$(cat "$prompt_file")"
keep=(HOME PATH LANG LC_ALL TMPDIR USER)
env_args=()
for v in "${keep[@]}"; do [[ -n "${!v:-}" ]] && env_args+=("$v=${!v}"); done
while IFS='=' read -r name _; do
    [[ "$name" =~ PROXY|proxy|CERT|CA_BUNDLE|CACERTS ]] && env_args+=("$name=${!name}")
done < <(env)
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work" && env -i "${env_args[@]}" timeout "${AGY_TIMEOUT:-300}" \
    "$agy" -p "$prompt" --model "$model" --sandbox --disable-slash-commands

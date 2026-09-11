#!/usr/bin/env bash
# Desktop stop only: both checkout scopes, no application runtime or port targeting.
set -u
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)" || exit 1
gemma_dir="$repo_dir/VibroAgent-Gemma"

stop_combined() (
  local arg stack result
  for arg in "$@"; do
    case "$arg" in
      --dry-run|--allow-legacy) ;;
      *) printf 'Usage: %s [--dry-run] [--allow-legacy]\n' "$0" >&2; return 2 ;;
    esac
  done
  # Check both entrypoints before stopping either stack. Never fall back to zip1.
  for stack in "$repo_dir" "$gemma_dir"; do
    if [[ ! -f "$stack/vibroagent.sh" || ! -f "$stack/vibrodiag_mcp_prototype/scripts/safe_pipeline_stop.py" ]]; then
      printf 'STOP INCOMPLETE: missing checkout stop entrypoint: %s\n' "$stack" >&2
      return 2
    fi
    mkdir -p "$stack/.run" || return $?
  done
  # Own BOTH existing control locks before any stop action. A busy Gemma stack
  # must not cause a partial Codec stop (and vice versa). Subshell exit releases
  # both locks before the desktop prompt; children never inherit the lock FDs.
  exec 9>"$repo_dir/.run/control.lock" || return $?
  flock --nonblock --conflict-exit-code 75 9 || return $?
  exec 8>"$gemma_dir/.run/control.lock" || return $?
  flock --nonblock --conflict-exit-code 75 8 || return $?
  for stack in "$repo_dir" "$gemma_dir"; do
    printf '\n== Safe-stop scope: %s ==\n' "$stack"
    /bin/bash "$stack/vibroagent.sh" --control-locked safe-stop "$@" 9>&- 8>&-
    result=$?
    if [[ "$result" != 0 ]]; then
      printf 'STOP INCOMPLETE: %s returned %s; no further stack was stopped.\n' "$stack" "$result" >&2
      return "$result"
    fi
  done
)

stop_combined "$@"
result=$?
if [[ "$result" == 75 ]]; then
  printf '\nAnother pipeline operation is in progress. No second stop signal was sent.\n'
fi
if [[ -t 0 ]]; then
  printf '\nPress Enter to close this window.\n'
  read -r _reply || true
fi
exit "$result"

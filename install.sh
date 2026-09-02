#!/usr/bin/env bash
# Interactive entry point for the two independently deployable VibroAgent stacks.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=""
DRY_RUN=0
NO_SPLASH=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [selection] [--dry-run] [--no-splash]

Run without a selection in a terminal for the interactive installer.

Selections:
  --codec-live       VibroAgent-Codec with six USB boards
  --codec-offline    VibroAgent-Codec with five-minute recorded replay
  --gemma-live       VibroAgent-Gemma with six USB boards
  --gemma-offline    VibroAgent-Gemma with 60-second LUMO replay
  --gemma-demo       VibroAgent-Gemma direct LUMO command-line demo

Options:
  --dry-run          Print the selected setup and run commands only
  --no-splash        Skip the short ASCII splash
  -h, --help         Show this help
EOF
}

set_target() {
  if [ -n "$TARGET" ]; then
    printf 'Choose only one deployment selection.\n' >&2
    exit 2
  fi
  TARGET="$1"
}

for arg in "$@"; do
  case "$arg" in
    --codec-live) set_target codec-live ;;
    --codec-offline) set_target codec-offline ;;
    --gemma-live) set_target gemma-live ;;
    --gemma-offline) set_target gemma-offline ;;
    --gemma-demo) set_target gemma-demo ;;
    --dry-run) DRY_RUN=1 ;;
    --no-splash) NO_SPLASH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BLUE=$'\033[38;5;27m'
  CYAN=$'\033[38;5;39m'
  MUTED=$'\033[38;5;244m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  BLUE=""; CYAN=""; MUTED=""; BOLD=""; RESET=""
fi

restore_cursor() { [ -t 1 ] && printf '\033[?25h'; }
trap restore_cursor EXIT

logo() {
  local spinner="${1:- }"
  local pulse="${2:-  &  }"
  local st_tone="${3:-$BLUE}"
  local q_tone="${4:-$CYAN}"
  printf '%s\n' \
    "${st_tone}        ${spinner}     _____ _______${RESET}           ${q_tone}____       ${spinner}${RESET}" \
    "${st_tone}             / ____|__   __|${RESET}         ${q_tone}/ __ \\${RESET}" \
    "${st_tone}            | (___    | |${RESET}           ${q_tone}| |  | |${RESET}" \
    "${st_tone}             \\___ \\   | |${RESET}           ${q_tone}| |  | |${RESET}" \
    "${st_tone}             ____) |  | |${RESET}           ${q_tone}| |__| |${RESET}" \
    "${st_tone}            |_____/   |_|${RESET}            ${q_tone}\\___\\_\\${RESET}" \
    "${st_tone}${BOLD}             STMicroelectronics${RESET} ${CYAN}${BOLD}${pulse}${RESET} ${q_tone}${BOLD}Qualcomm${RESET}" \
    "${CYAN}          VIBROAGENT // IQ-9075 // HEXAGON HTP${RESET}"
}

splash() {
  local spinners=('|' '/' '-' $'\\' '|' '/' '-' $'\\')
  local pulses=('&....' '.&...' '..&..' '...&.' '....&' '...&.' '..&..' '.&...')
  local frame st_tone q_tone
  printf '\033[?25l'
  for frame in "${!spinners[@]}"; do
    if [ "$frame" -lt 3 ]; then
      st_tone="$BLUE$BOLD"; q_tone="$MUTED"
    elif [ "$frame" -lt 6 ]; then
      st_tone="$MUTED"; q_tone="$CYAN$BOLD"
    else
      st_tone="$BLUE"; q_tone="$CYAN"
    fi
    printf '\033[2J\033[H'
    logo "${spinners[$frame]}" "${pulses[$frame]}" "$st_tone" "$q_tone"
    printf '\n%s              Preparing installer...%s\n' "$MUTED" "$RESET"
    sleep 0.065
  done
  printf '\033[?25h'
}

choose_target() {
  local selected=0 key rest index
  local targets=(codec-live codec-offline gemma-live gemma-offline gemma-demo exit)
  local labels=(
    'VibroAgent-Codec  | Live six-board deployment'
    'VibroAgent-Codec  | Offline five-minute replay'
    'VibroAgent-Gemma  | Live six-board deployment'
    'VibroAgent-Gemma  | Offline LUMO web pipeline'
    'VibroAgent-Gemma  | Direct LUMO CLI demo'
    'Exit'
  )
  local notes=(
    'Discrete codec tokens + Qwen3-4B; requires six STWIN.box boards.'
    'Same Codec pipeline over downloaded immutable recordings; no boards.'
    '84 continuous vibration embeddings + Gemma Q8; requires six boards.'
    'Full web UI over included .dat files; scheduled Target 3/5 LUMO events.'
    'Run one included LUMO window directly through encoder and model.'
    'Leave the repository unchanged.'
  )

  printf '\033[?25l'
  while true; do
    printf '\033[2J\033[H'
    logo
    printf '\n%sWelcome to the combined VibroAgent installer.%s\n' "$BOLD" "$RESET"
    printf '%sChoose a deployment with Up/Down and press Enter. Press q to exit.%s\n\n' "$MUTED" "$RESET"
    for index in "${!labels[@]}"; do
      if [ "$index" -eq "$selected" ]; then
        printf '%s  > %-57s%s\n' "$CYAN$BOLD" "${labels[$index]}" "$RESET"
        printf '%s      %s%s\n' "$MUTED" "${notes[$index]}" "$RESET"
      else
        printf '    %-57s\n' "${labels[$index]}"
        printf '      %s%s%s\n' "$MUTED" "${notes[$index]}" "$RESET"
      fi
    done

    IFS= read -rsn1 key
    case "$key" in
      $'\033')
        rest=""
        IFS= read -rsn2 -t 0.1 rest || true
        case "$rest" in
          '[A') selected=$((selected > 0 ? selected - 1 : ${#labels[@]} - 1)) ;;
          '[B') selected=$((selected + 1 < ${#labels[@]} ? selected + 1 : 0)) ;;
        esac
        ;;
      '') TARGET="${targets[$selected]}"; break ;;
      q|Q) TARGET=exit; break ;;
    esac
  done
  printf '\033[?25h\033[2J\033[H'
}

describe_target() {
  case "$TARGET" in
    codec-live)
      LABEL='VibroAgent-Codec / live boards'
      SETUP=("$ROOT/setup.sh")
      NEXT=("$ROOT/vibroagent.sh" start)
      ;;
    codec-offline)
      LABEL='VibroAgent-Codec / offline replay'
      SETUP=("$ROOT/setup.sh" --offline)
      NEXT=("$ROOT/vibroagent.sh" start)
      ;;
    gemma-live)
      LABEL='VibroAgent-Gemma / live boards'
      SETUP=("$ROOT/VibroAgent-Gemma/setup.sh" --live)
      NEXT=("$ROOT/VibroAgent-Gemma/vibroagent.sh" start)
      ;;
    gemma-offline)
      LABEL='VibroAgent-Gemma / offline LUMO replay'
      SETUP=("$ROOT/VibroAgent-Gemma/setup.sh" --offline)
      NEXT=("$ROOT/VibroAgent-Gemma/vibroagent.sh" start)
      ;;
    gemma-demo)
      LABEL='VibroAgent-Gemma / direct LUMO demo'
      SETUP=("$ROOT/VibroAgent-Gemma/setup.sh" --demo)
      NEXT=("$ROOT/VibroAgent-Gemma/vibroagent.sh" demo target_3)
      ;;
    exit) printf 'Installer closed.\n'; exit 0 ;;
    *) printf 'Internal installer selection error: %s\n' "$TARGET" >&2; exit 2 ;;
  esac
}

print_command() {
  printf ' %q' "$@"
  printf '\n'
}

if [ -z "$TARGET" ]; then
  if [ ! -t 0 ] || [ ! -t 1 ]; then
    printf 'Interactive selection needs a terminal; pass one selection flag.\n\n' >&2
    usage >&2
    exit 2
  fi
  [ "$NO_SPLASH" -eq 1 ] || splash
  choose_target
fi

describe_target
printf '%sSelected:%s %s\n' "$BOLD" "$RESET" "$LABEL"
printf '%sSetup:%s' "$MUTED" "$RESET"; print_command "${SETUP[@]}"
printf '%sRun:%s  ' "$MUTED" "$RESET"; print_command "${NEXT[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

printf '\n%sInstalling %s...%s\n\n' "$BOLD" "$LABEL" "$RESET"
"${SETUP[@]}"
status=$?
if [ "$status" -ne 0 ]; then
  printf '\nInstallation failed with exit code %s. Review the output above.\n' "$status" >&2
  exit "$status"
fi

printf '\n%sInstallation complete.%s Start the selected pipeline with:\n' "$BOLD" "$RESET"
print_command "${NEXT[@]}"
if [ "$TARGET" = gemma-demo ]; then
  printf 'Use target_5 instead of target_3 for the second bundled LUMO condition.\n'
fi

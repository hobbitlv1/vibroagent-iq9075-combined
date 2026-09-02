#!/usr/bin/env bash
# Interactive entry point for the two independently deployable VibroAgent stacks.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=""
DRY_RUN=0
NO_SPLASH=0
NO_ANIMATION=0
ASCII=0
SETUP_PID=""
ALT_SCREEN=0
SUDO_KEEPALIVE=0

usage() {
  cat <<'USAGE'
Usage: ./install.sh [selection] [--dry-run] [--no-splash] [--no-animation]

Run without a selection in a terminal for the interactive installer.

Selections:
  --codec-live         VibroAgent-Codec with six USB boards
  --codec-offline      VibroAgent-Codec with five-minute recorded replay
  --gemma-live         VibroAgent-Gemma with six USB boards
  --gemma-offline      VibroAgent-Gemma with 60-second LUMO replay
  --gemma-demo         VibroAgent-Gemma direct LUMO command-line demo

Options:
  --dry-run            Print the selected setup and run commands only
  --no-splash          Skip the short brand animation
  --no-animation       Show normal setup output instead of live progress
  --ascii              Use plain ASCII glyphs instead of Unicode
  --preview-animation  Preview the live progress without changing anything
  -h, --help           Show this help

Environment:
  NO_COLOR=1           Disable colours
  VIBRO_NO_ANIMATION=1 Same as --no-animation
USAGE
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
    --no-animation) NO_ANIMATION=1 ;;
    --ascii) ASCII=1 ;;
    --preview-animation|--preview) set_target preview ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

# --- Terminal capabilities ----------------------------------------------------

IS_TTY=0
[ -t 1 ] && IS_TTY=1
COLS=80
[ "$IS_TTY" -eq 1 ] && COLS="$(tput cols 2>/dev/null || echo 80)"
case "$COLS" in ''|*[!0-9]*) COLS=80 ;; esac
[ "$COLS" -ge 48 ] || COLS=48
WIDTH=$((COLS > 100 ? 100 : COLS))
ROWS=24
[ "$IS_TTY" -eq 1 ] && ROWS="$(tput lines 2>/dev/null || echo 24)"
case "$ROWS" in ''|*[!0-9]*) ROWS=24 ;; esac
COMPACT=0
[ "$ROWS" -ge 32 ] || COMPACT=1

UNICODE=0
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *[Uu][Tt][Ff]-8*|*[Uu][Tt][Ff]8*) UNICODE=1 ;;
esac
[ "$ASCII" -eq 1 ] && UNICODE=0

if [ "$UNICODE" -eq 1 ]; then
  SPIN=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  G_OK='✔'; G_FAIL='✘'; G_PTR='▸'; G_FILL='━'; G_EMPTY='┈'; G_SEP='·'
  LEVELS=('▁' '▂' '▃' '▄' '▅' '▆' '▇' '█')
  B_H='─'; B_V='│'; B_TL='╭'; B_TR='╮'; B_BL='╰'; B_BR='╯'
else
  SPIN=('|' '/' '-' '\')
  G_OK='+'; G_FAIL='x'; G_PTR='>'; G_FILL='='; G_EMPTY='.'; G_SEP='-'
  LEVELS=('_' '.' '-' '~' '=' '+' '*' '#')
  B_H='-'; B_V='|'; B_TL='+'; B_TR='+'; B_BL='+'; B_BR='+'
fi

if [ "$IS_TTY" -eq 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then
  BLUE=$'\033[38;5;27m'; BLUE_DIM=$'\033[38;5;24m'
  CYAN=$'\033[38;5;39m'; CYAN_HI=$'\033[38;5;51m'
  QUALCOMM=$'\033[38;2;50;83;220m'
  MUTED=$'\033[38;5;244m'; DIM=$'\033[38;5;238m'; RED=$'\033[38;5;203m'
  BOLD=$'\033[1m'; RESET=$'\033[0m'
else
  BLUE=""; BLUE_DIM=""; CYAN=""; CYAN_HI=""; QUALCOMM=""
  MUTED=""; DIM=""; RED=""; BOLD=""; RESET=""
fi
CLR=""
[ "$IS_TTY" -eq 1 ] && CLR=$'\033[2K'
COLOR=0
[ -n "$RESET" ] && COLOR=1

cleanup() {
  if [ -n "$SETUP_PID" ] && kill -0 "$SETUP_PID" 2>/dev/null; then
    kill "$SETUP_PID" 2>/dev/null || true
    wait "$SETUP_PID" 2>/dev/null || true
  fi
  if [ "$IS_TTY" -eq 1 ]; then
    [ "$ALT_SCREEN" -eq 1 ] && printf '\033[?1049l'
    printf '\033[?25h'
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

enter_alt() { [ "$ALT_SCREEN" -eq 1 ] && return; printf '\033[?1049h\033[H\033[2J\033[?25l'; ALT_SCREEN=1; }
leave_alt() { [ "$ALT_SCREEN" -eq 0 ] && return; printf '\033[?1049l\033[?25h'; ALT_SCREEN=0; }

# Sleep for $1 seconds; return 0 when a key was pressed (used to skip the splash).
frame() {
  if [ -t 0 ]; then
    IFS= read -rsn1 -t "$1" _ 2>/dev/null && return 0
  else
    sleep "$1"
  fi
  return 1
}

drain_input() { while IFS= read -rsn1 -t 0.01 _ 2>/dev/null; do :; done; }

fmt_time() { printf '%d:%02d' $(($1 / 60)) $(($1 % 60)); }

# --- Brand marks ---------------------------------------------------------------

# grad <i> <n> <variant: 0 normal, 1 bright, 2 dim> -> C
# Cell colour along the bar: ST light blue on the ST side, blending into the
# Qualcomm violet-blue as it approaches the Q mark.
grad() {
  if [ "$COLOR" -eq 0 ]; then C=''; return; fi
  local i="$1" n="$2" r g b
  r=$(( 60 + (50 - 60) * i / (n - 1) ))
  g=$(( 180 + (83 - 180) * i / (n - 1) ))
  b=$(( 229 + (220 - 229) * i / (n - 1) ))
  case "$3" in
    1) r=$(( r + (255 - r) * 55 / 100 )); g=$(( g + (255 - g) * 55 / 100 )); b=$(( b + (255 - b) * 55 / 100 )) ;;
    2) r=$(( r * 40 / 100 )); g=$(( g * 40 / 100 )); b=$(( b * 40 / 100 )) ;;
  esac
  C=$'\033[38;2;'"$r;$g;$b"m
}

# One period of a sine wave quantised to the eight sparkline levels.
WAVE=(4 5 6 7 7 7 6 5 4 2 1 0 0 0 1 2)

# wave_cell <i> <n> <tick> <amp 0-4> <variant> : append one waveform cell to WROW.
# The signal travels from the ST side toward the Q mark as tick advances.
wave_cell() {
  local i="$1" n="$2" tick="$3" amp="$4" lv C
  if [ "$amp" -le 0 ]; then WROW+="$DIM${LEVELS[0]}$RESET"; return; fi
  lv=$(( WAVE[(i - tick + 1600) % 16] * amp / 4 ))
  grad "$i" "$n" "$5"
  WROW+="$C${LEVELS[$lv]}$RESET"
}

# wave_row <width> <tick> <pulse|steady> <pos|amp> -> WROW
wave_row() {
  local w="$1" tick="$2" mode="$3" arg="$4" i d amp
  WROW=''
  for ((i = 0; i < w; i++)); do
    if [ "$mode" = pulse ]; then
      d=$(( i - arg )); [ "$d" -ge 0 ] || d=$(( -d ))
      if [ "$d" -eq 0 ]; then amp=4; elif [ "$d" -le 1 ]; then amp=3
      elif [ "$d" -le 3 ]; then amp=2; elif [ "$d" -le 6 ]; then amp=1; else amp=0; fi
      wave_cell "$i" "$w" "$tick" "$amp" $(( d == 0 ? 1 : 0 ))
    else
      wave_cell "$i" "$w" "$tick" "$arg" 0
    fi
  done
}

ST_ART=(
  '     _____ _______'
  '    / ____|__   __|'
  '   | (___    | |'
  '    \___ \   | |'
  '    ____) |  | |'
  '   |_____/   |_|'
)
Q_ART=(
  '  ____'
  ' / __ \'
  '| |  | |'
  '| |  | |'
  '| |__| |'
  ' \___\_\'
)
TAGLINE='VIBROAGENT // IQ-9075 // HEXAGON HTP'
LOGO_W=37
PAD=$(( (WIDTH - LOGO_W) / 2 ))
[ "$PAD" -ge 2 ] || PAD=2

# draw_logo <st_cols> <q_cols> <amp_tone> <caption: 0 hidden, 1 muted, 2 brand> <tagline_chars> <wave_row>
draw_logo() {
  local st_n="$1" q_n="$2" amp_tone="$3" cap="$4" tag_n="$5" wave="${6:-}"
  local i pad
  printf -v pad '%*s' "$PAD" ''
  for i in 0 1 2 3 4 5; do
    printf '%s%s%s%-19.19s%s        %s%s%s\n' "$CLR" "$pad" "$BLUE$BOLD" \
      "${ST_ART[$i]:0:$st_n}" "$RESET" "$QUALCOMM$BOLD" "${Q_ART[$i]:0:$q_n}" "$RESET"
  done
  case "$cap" in
    0) printf '%s\n' "$CLR" ;;
    1) printf '%s%s    %sSTMicroelectronics%s   %s&%s   %sQualcomm%s\n' "$CLR" "$pad" \
         "$MUTED" "$RESET" "$amp_tone" "$RESET" "$MUTED" "$RESET" ;;
    *) printf '%s%s    %sSTMicroelectronics%s   %s&%s   %sQualcomm%s\n' "$CLR" "$pad" \
         "$BLUE$BOLD" "$RESET" "$amp_tone" "$RESET" "$QUALCOMM$BOLD" "$RESET" ;;
  esac
  printf '%s%s%s\n' "$CLR" "$pad" "$wave"
  printf '%s%s %s%s%s\n' "$CLR" "$pad" "$CYAN" "${TAGLINE:0:$tag_n}" "$RESET"
}

# logo_static [tick]: full marks with a calm travelling wave underneath.
logo_static() {
  wave_row "$LOGO_W" "${1:-0}" steady 2
  draw_logo 99 99 "$CYAN_HI$BOLD" 2 99 "$WROW"
}

# Brand reveal: the ST mark wipes in, the Q mark answers, the ampersand pulses
# between them, and the tagline types out. Any key skips ahead.
splash() {
  local n tone wt=0
  enter_alt
  # splash_wave: a vibration pulse travels from the ST side to the Q mark, then
  # the signal settles into a steady wave.
  splash_wave() {
    wt=$((wt + 1))
    if [ $((wt * 2)) -lt "$LOGO_W" ]; then wave_row "$LOGO_W" "$wt" pulse $((wt * 2))
    else wave_row "$LOGO_W" "$wt" steady 3; fi
  }
  for n in 3 6 9 12 15 19; do
    splash_wave; printf '\033[H'; draw_logo "$n" 0 "$DIM" 0 0 "$WROW"
    frame 0.045 && { finish_splash; return; }
  done
  for n in 2 4 6 8; do
    splash_wave; printf '\033[H'; draw_logo 99 "$n" "$DIM" 0 0 "$WROW"
    frame 0.045 && { finish_splash; return; }
  done
  splash_wave; printf '\033[H'; draw_logo 99 99 "$DIM" 1 0 "$WROW"
  frame 0.12 && { finish_splash; return; }
  for tone in "$MUTED" "$BLUE" "$CYAN" "$CYAN_HI$BOLD" "$CYAN" "$BLUE_DIM" "$CYAN" "$CYAN_HI$BOLD"; do
    splash_wave; printf '\033[H'; draw_logo 99 99 "$tone" 2 0 "$WROW"
    frame 0.07 && { finish_splash; return; }
  done
  for n in 4 8 12 16 20 24 28 32 36; do
    splash_wave; printf '\033[H'; draw_logo 99 99 "$CYAN_HI$BOLD" 2 "$n" "$WROW"
    frame 0.035 && { finish_splash; return; }
  done
  frame 0.25 || true
  finish_splash
}

finish_splash() {
  printf '\033[H'; logo_static
  drain_input
}

# --- Menu ----------------------------------------------------------------------

choose_target() {
  local selected=0 key rest index group last_group hint tick=0 rc
  local targets=(codec-live codec-offline gemma-live gemma-offline gemma-demo exit)
  local groups=(codec codec gemma gemma gemma '')
  local labels=(
    'Live six-board deployment'
    'Offline five-minute replay'
    'Live six-board deployment'
    'Offline LUMO web pipeline'
    'Direct LUMO CLI demo'
    'Exit installer'
  )
  local notes=(
    'Discrete codec tokens + Qwen3-4B. Requires six STWIN.box boards.'
    'Same Codec pipeline over downloaded immutable recordings. No boards.'
    '84 continuous vibration embeddings + Gemma Q8. Requires six boards.'
    'Full web UI over included .dat files with scheduled Target 3/5 LUMO events.'
    'Run one included LUMO window directly through encoder and model.'
    'Leave the repository unchanged.'
  )
  local cmds=(
    './setup.sh'
    './setup.sh --offline'
    './VibroAgent-Gemma/setup.sh --live'
    './VibroAgent-Gemma/setup.sh --offline'
    './VibroAgent-Gemma/setup.sh --demo'
    ''
  )

  enter_alt
  while true; do
    printf '\033[H'
    logo_static "$tick"
    if [ "$COMPACT" -eq 1 ]; then
      printf '%s  %sChoose a deployment%s\n' "$CLR" "$BOLD" "$RESET"
    else
      printf '%s\n%s  %sChoose a deployment%s\n%s\n' "$CLR" "$CLR" "$BOLD" "$RESET" "$CLR"
    fi
    last_group=''
    for index in "${!labels[@]}"; do
      group="${groups[$index]}"
      if [ "$group" != "$last_group" ]; then
        [ -n "$last_group" ] && [ "$COMPACT" -eq 0 ] && printf '%s\n' "$CLR"
        case "$group" in
          codec) printf '%s  %sVibroAgent-Codec%s  %sRVQ codec tokens %s Qwen3-4B%s\n' \
                   "$CLR" "$BLUE$BOLD" "$RESET" "$MUTED" "$G_SEP" "$RESET" ;;
          gemma) printf '%s  %sVibroAgent-Gemma%s  %scontinuous encoder %s Gemma 4 E2B Q8%s\n' \
                   "$CLR" "$QUALCOMM$BOLD" "$RESET" "$MUTED" "$G_SEP" "$RESET" ;;
          *) [ "$COMPACT" -eq 1 ] || printf '%s\n' "$CLR" ;;
        esac
        last_group="$group"
      fi
      if [ "$index" -lt 5 ]; then hint="$((index + 1))"; else hint='q'; fi
      if [ "$index" -eq "$selected" ]; then
        printf '%s  %s%s %s%s  %s%-30s%s\n' "$CLR" "$CYAN_HI" "$G_PTR" "$hint" "$RESET" \
          "$CYAN_HI$BOLD" "${labels[$index]}" "$RESET"
        [ "$COMPACT" -eq 1 ] || printf '%s        %s\n' "$CLR" "${notes[$index]}"
      else
        printf '%s    %s%s%s  %-30s\n' "$CLR" "$DIM" "$hint" "$RESET" "${labels[$index]}"
        [ "$COMPACT" -eq 1 ] || printf '%s        %s%s%s\n' "$CLR" "$MUTED" "${notes[$index]}" "$RESET"
      fi
    done
    printf '%s\n' "$CLR"
    [ "$COMPACT" -eq 1 ] && printf '%s  %s\n' "$CLR" "${notes[$selected]}"
    if [ -n "${cmds[$selected]}" ]; then
      printf '%s  %sEnter runs%s  %s\n' "$CLR" "$MUTED" "$RESET" "${cmds[$selected]}"
    else
      printf '%s  %sEnter closes the installer%s\n' "$CLR" "$MUTED" "$RESET"
    fi
    printf '%s  %sUp/Down or j/k move  %s  1-5 jump  %s  Enter select  %s  q quit%s\n' \
      "$CLR" "$DIM" "$G_SEP" "$G_SEP" "$G_SEP" "$RESET"
    printf '\033[J'

    IFS= read -rsn1 -t 0.12 key; rc=$?
    if [ "$rc" -gt 128 ]; then tick=$((tick + 1)); continue; fi
    [ "$rc" -eq 0 ] || { TARGET=exit; break; }
    case "$key" in
      $'\033')
        rest=""
        IFS= read -rsn2 -t 0.1 rest || true
        case "$rest" in
          '[A'|'OA') selected=$((selected > 0 ? selected - 1 : ${#labels[@]} - 1)) ;;
          '[B'|'OB') selected=$((selected + 1 < ${#labels[@]} ? selected + 1 : 0)) ;;
          '[H'|'OH') selected=0 ;;
          '[F'|'OF') selected=$((${#labels[@]} - 1)) ;;
          '[1') selected=0; IFS= read -rsn1 -t 0.05 _ || true ;;
          '[4') selected=$((${#labels[@]} - 1)); IFS= read -rsn1 -t 0.05 _ || true ;;
        esac
        ;;
      k|K) selected=$((selected > 0 ? selected - 1 : ${#labels[@]} - 1)) ;;
      j|J) selected=$((selected + 1 < ${#labels[@]} ? selected + 1 : 0)) ;;
      [1-5]) selected=$((key - 1)) ;;
      ''|' ') TARGET="${targets[$selected]}"; break ;;
      q|Q) TARGET=exit; break ;;
    esac
  done
  leave_alt
}

# --- Live progress -------------------------------------------------------------

# Read log output added since the previous poll and process every complete
# line: "== " or "==== " markers open a new phase, other lines feed the ticker.
# Sets/uses the caller's phase state (current, phase, phase_started, step_n,
# step_total, counter, detail) through dynamic scoping.
LOG_OFFSET=0
poll_log() {
  local chunk rest line text nl=$'\n' cr=$'\r'
  local LC_ALL=C
  chunk="$(tail -c +$((LOG_OFFSET + 1)) "$1" 2>/dev/null; printf x)"
  chunk="${chunk%x}"
  rest="${chunk##*[$nl$cr]}"
  chunk="${chunk%"$rest"}"
  [ -n "$chunk" ] || return 0
  LOG_OFFSET=$((LOG_OFFSET + ${#chunk}))
  while IFS= read -r line; do
    text="${line:1}"
    case "${line:0:1}" in
      1|0)
        [ "$text" != "$current" ] || continue
        [ "$phase" -gt 0 ] && phase_done "$current" $((SECONDS - phase_started))
        current="$text"
        phase=$((phase + 1))
        phase_started=$SECONDS
        detail=''
        if [ "${line:0:1}" = 1 ] && [[ "$text" =~ \[([0-9]+)/([0-9]+)\] ]]; then
          step_n="${BASH_REMATCH[1]}"; step_total="${BASH_REMATCH[2]}"
          counter="$step_n/$step_total"
        fi
        ;;
      L) detail="$text" ;;
    esac
  done < <(printf '%s' "$chunk" | tr '\r' '\n' | awk '
    { gsub(/\033\[[0-9;?]*[A-Za-z]/, ""); gsub(/\t/, " ") }
    /^===+ / { m=$0; sub(/^=+ +/, "", m); sub(/ +=+$/, "", m); print "1" m; last=""; next }
    /^== /   { m=$0; sub(/^=+ +/, "", m); print "0" m; last=""; next }
    /[^ ]/   { last=$0 }
    END { if (last != "") print "L" last }')
}

# build_bar <width> <tick> <sweep_pos> <sweep_dir> <step_n> <step_total> -> BAR
# Waveform bar: energised cells carry the signal, idle cells stay flat.
build_bar() {
  local w="$1" tick="$2" pos="$3" dir="$4" n="$5" total="$6"
  local i d done_cells cur_end head amp
  WROW=''
  if [ "$total" -gt 0 ]; then
    done_cells=$(( (n - 1) * w / total ))
    cur_end=$(( n * w / total ))
    [ "$cur_end" -gt "$done_cells" ] || cur_end=$((done_cells + 1))
    [ "$cur_end" -le "$w" ] || cur_end="$w"
    # The head keeps its position across steps: it flows on through finished
    # cells into the new segment instead of restarting the animation.
    [ "$HEAD" -ge 0 ] || HEAD="$pos"
    [ $((tick % 2)) -ne 0 ] || HEAD=$((HEAD + 1))
    [ "$HEAD" -lt "$cur_end" ] || HEAD="$done_cells"
    head="$HEAD"
    for ((i = 0; i < w; i++)); do
      if [ "$i" -lt "$done_cells" ]; then wave_cell "$i" "$w" "$tick" 4 0
      elif [ "$i" -eq "$head" ]; then wave_cell "$i" "$w" "$tick" 4 1
      elif [ "$i" -lt "$cur_end" ]; then wave_cell "$i" "$w" "$tick" 2 2
      else wave_cell "$i" "$w" "$tick" 0 0
      fi
    done
  else
    for ((i = 0; i < w; i++)); do
      d=$(( i - pos )); [ "$d" -ge 0 ] || d=$(( -d ))
      if [ "$d" -eq 0 ]; then amp=4; elif [ "$d" -le 1 ]; then amp=3
      elif [ "$d" -le 3 ]; then amp=2; elif [ "$d" -le 5 ]; then amp=1; else amp=0; fi
      wave_cell "$i" "$w" "$tick" "$amp" $(( d == 0 ? 1 : 0 ))
    done
  fi
  BAR="$WROW"
}

# phase_line <glyph_tone> <glyph> <text> <seconds>: one finished checklist row.
phase_line() {
  local tstr
  tstr="$(fmt_time "$4")"
  printf '%s  %s%s%s %s  %s%s%s\n' "$CLR" "$1" "$2" "$RESET" \
    "${3:0:$((WIDTH - 8 - ${#tstr}))}" "$MUTED" "$tstr" "$RESET"
}
phase_done() { phase_line "$CYAN" "$G_OK" "$1" "$2"; }

run_setup() {
  local log="$ROOT/.run/install.log"
  mkdir -p "$ROOT/.run"
  : > "$log"

  if [ "$IS_TTY" -eq 0 ] || [ "$NO_ANIMATION" -eq 1 ] || [ -n "${VIBRO_NO_ANIMATION:-}" ]; then
    "$@" 2>&1 | tee "$log"
    return "${PIPESTATUS[0]}"
  fi

  local bar_w=24 started=$SECONDS phase_started=$SECONDS
  local tick=0 pos=0 dir=1 phase=0 status completed
  HEAD=-1
  local current='Starting setup' detail=''
  LOG_OFFSET=0
  local step_n=0 step_total=0 counter='' tstr avail line1 line2 spin

  "$@" >>"$log" 2>&1 &
  SETUP_PID=$!
  printf '\033[?25l'
  while kill -0 "$SETUP_PID" 2>/dev/null; do
    [ $((tick % 3)) -ne 0 ] || poll_log "$log"

    spin="${SPIN[tick % ${#SPIN[@]}]}"
    build_bar "$bar_w" "$tick" "$pos" "$dir" "$step_n" "$step_total"
    tstr="$(fmt_time $((SECONDS - phase_started)))"
    avail=$((WIDTH - 7 - ${#tstr}))
    printf -v line1 '%s  %s%s%s %s%s%s  %s%s%s' "$CLR" "$CYAN_HI" "$spin" "$RESET" \
      "$BOLD" "${current:0:$avail}" "$RESET" "$MUTED" "$tstr" "$RESET"
    tstr="$(fmt_time $((SECONDS - started)))"
    avail=$((WIDTH - 13 - bar_w - ${#counter} - ${#tstr}))
    [ "$avail" -gt 0 ] || avail=0
    printf -v line2 '%s  %s%sST%s %s %s%sQ%s  %s%s %s%s  %s%s%s' "$CLR" \
      "$BLUE" "$BOLD" "$RESET" "$BAR" "$QUALCOMM" "$BOLD" "$RESET" \
      "$MUTED" "$counter" "$tstr" "$RESET" "$DIM" "${detail:0:$avail}" "$RESET"
    printf '%s\n%s\r\033[1A' "$line1" "$line2"

    sleep 0.08
    tick=$((tick + 1))
    pos=$((pos + dir))
    if [ "$pos" -ge $((bar_w - 1)) ] || [ "$pos" -le 0 ]; then dir=$((-dir)); fi
    if [ "$SUDO_KEEPALIVE" -eq 1 ] && [ $((tick % 600)) -eq 0 ]; then
      sudo -n -v 2>/dev/null || true
    fi
  done
  wait "$SETUP_PID"; status=$?
  SETUP_PID=""
  printf '%s\n%s\r\033[1A\033[?25h' "$CLR" "$CLR"
  poll_log "$log"

  if [ "$status" -ne 0 ]; then
    phase_line "$RED" "$G_FAIL" "$current" $((SECONDS - phase_started))
    printf '\n  %sInstallation failed%s with exit code %s after %s.\n' "$RED$BOLD" "$RESET" \
      "$status" "$(fmt_time $((SECONDS - started)))"
    printf '  %sLast output:%s\n' "$MUTED" "$RESET"
    tail -n 20 "$log" | tr '\r' '\n' | awk 'NF' | tail -n 20 | while IFS= read -r line; do
      printf '  %s%s %s%s\n' "$DIM" "$B_V" "${line:0:$((WIDTH - 4))}" "$RESET"
    done
    printf '  %sFull log%s  %s\n' "$MUTED" "$RESET" "$log"
    return "$status"
  fi
  [ "$phase" -gt 0 ] && phase_done "$current" $((SECONDS - phase_started))
  if [ "$step_total" -gt 0 ]; then completed="$step_total"; else completed="$phase"; fi
  printf '\n  %s%s Installed%s %s  %s%d steps %s %s%s\n' "$CYAN_HI$BOLD" "$G_OK" "$RESET" \
    "$LABEL" "$MUTED" "$completed" "$G_SEP" "$(fmt_time $((SECONDS - started)))" "$RESET"
  printf '  %sLog%s  %s\n' "$MUTED" "$RESET" "$log"
}

preview_installation() {
  step() {
    printf '==== [%s/5] %s ====\n' "$1" "$2"; shift 2
    local line
    for line in "$@"; do printf '%s\n' "$line"; sleep 0.35; done
  }
  echo '==== data-source mode: preview (nothing is changed) ===='; sleep 0.8
  step 1 'Checking host and available storage' 'Host: IQ-9075 / aarch64' 'Free space on /: 118 GB'
  step 2 'Verifying SDK and patch revisions' '== fetching stock STDATALOG-PYSDK v1.3.0' \
    'Receiving objects: 100% (2143/2143)' '== applying VibroAgent SDK patches (sdk_patches/overlay/)' \
    'patched stdatalog_core/HSD_utils/dtm.py'
  step 3 'Preparing Python environments' 'Resolved 84 packages in 1.2s' \
    'Downloading torch (212.4 MiB)' 'Installed 84 packages in 9.8s'
  step 4 'Verifying model release assets' 'codes_v3-q8.gguf: sha256 OK' 'lumo_windows.tar: sha256 OK'
  step 5 'Running deployment health checks' 'encoder warm-up: 0.42 s' 'HTP backend: online'
  echo '==== setup complete ===='; sleep 0.6
}

# --- Selection -----------------------------------------------------------------

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

# Command relative to the repository root, for the final "next step" panel.
rel_command() {
  local parts=("$@")
  parts[0]="./${parts[0]#"$ROOT"/}"
  local out
  printf -v out '%q ' "${parts[@]}"
  printf '%s' "${out% }"
}

# box <title> <line>... : bordered panel, falls back to plain lines when too wide.
box() {
  local title="$1"; shift
  local line inner=0 w rule
  for line in "$@"; do [ "${#line}" -gt "$inner" ] && inner="${#line}"; done
  w=$((inner + 4))
  if [ "$w" -gt $((WIDTH - 2)) ]; then
    printf '  %s%s%s\n' "$BOLD" "$title" "$RESET"
    for line in "$@"; do printf '    %s\n' "$line"; done
    return
  fi
  printf -v rule '%*s' $((w - ${#title} - 4)) ''
  rule="${rule// /$B_H}"
  printf '  %s%s%s %s%s%s %s%s%s\n' "$DIM" "$B_TL$B_H" "$RESET" "$BOLD" "$title" "$RESET" "$DIM" "$rule$B_TR" "$RESET"
  for line in "$@"; do
    printf '  %s%s%s  %-*s  %s%s%s\n' "$DIM" "$B_V" "$RESET" "$inner" "$line" "$DIM" "$B_V" "$RESET"
  done
  printf -v rule '%*s' $((w - 2)) ''
  printf '  %s%s%s%s%s\n' "$DIM" "$B_BL" "${rule// /$B_H}" "$B_BR" "$RESET"
}

# --- Main ----------------------------------------------------------------------

if [ -z "$TARGET" ]; then
  if [ ! -t 0 ] || [ ! -t 1 ]; then
    printf 'Interactive selection needs a terminal; pass one selection flag.\n\n' >&2
    usage >&2
    exit 2
  fi
  [ "$NO_SPLASH" -eq 1 ] || splash
  choose_target
  logo_static
  printf '\n'
elif [ "$IS_TTY" -eq 1 ]; then
  printf '  %sST%s %s&%s %sQualcomm%s  %sVibroAgent installer%s\n\n' \
    "$BLUE$BOLD" "$RESET" "$CYAN_HI" "$RESET" "$QUALCOMM$BOLD" "$RESET" "$MUTED" "$RESET"
fi

if [ "$TARGET" = preview ]; then
  if [ "$IS_TTY" -eq 0 ]; then
    printf 'Animation preview needs a terminal.\n' >&2
    exit 2
  fi
  LABEL='animation preview'
  printf '  %sInstalling%s %s\n\n' "$BOLD" "$RESET" "$LABEL"
  run_setup preview_installation
  exit $?
fi

describe_target
printf '  %sSelected%s  %s\n' "$MUTED" "$RESET" "$LABEL"
printf '  %sSetup%s    ' "$MUTED" "$RESET"; print_command "${SETUP[@]}"
printf '  %sRun%s      ' "$MUTED" "$RESET"; print_command "${NEXT[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
  exit 0
fi

printf '\n'
if [[ "$TARGET" == *-live ]] && [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  printf '  Live setup requires administrator access.\n'
  sudo -v || exit $?
  SUDO_KEEPALIVE=1
fi
printf '  %sInstalling%s %s\n\n' "$BOLD" "$RESET" "$LABEL"
run_setup "${SETUP[@]}"
status=$?
if [ "$status" -ne 0 ]; then
  printf '\nInstallation failed with exit code %s. Review the output above.\n' "$status" >&2
  exit "$status"
fi

printf '\n'
next_lines=()
[ "$PWD" = "$ROOT" ] || next_lines+=("cd $(printf '%q' "$ROOT")")
next_lines+=("$(rel_command "${NEXT[@]}")")
if [ "$TARGET" = gemma-demo ]; then
  next_lines+=('Use target_5 instead of target_3 for the second bundled LUMO condition.')
fi
box 'Next' "${next_lines[@]}"

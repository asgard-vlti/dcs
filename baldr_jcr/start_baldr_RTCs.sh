#!/usr/bin/env bash
set -euo pipefail

# ---- Usage ----
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <mode> <mask> [beam_list]"
  echo "  <mode>      : required"
  echo "  <mask>      : required"
  echo "  [beam_list] : optional comma-separated (e.g. 1,2,4). Default: 1,2,3,4"
  exit 1
fi
MODE="$1"
MASK="$2"
BEAM_INPUT="${3:-1,2,3,4}"

cd /home/asg/Progs/repos/dcs/baldr

if [[ -z "${DISPLAY:-}" ]]; then
  echo "Error: No DISPLAY found (GUI not available)."
  echo "If on SSH, try:  ssh -X <host>    and verify 'xterm' can open."
  exit 1
fi

IFS=',' read -ra BEAMS <<< "$BEAM_INPUT"

# ---- Beam→Port mapping ----
declare -A PORTS=([1]=6662 [2]=6663 [3]=6664 [4]=6665)

# ---- Launchers ----
launch_with() {
  local term="$1"; shift
  local title="$1"; shift
  local cmd="$*"
  case "$term" in
    gnome-terminal)
      gnome-terminal --title="$title" -- bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    kgx)
      kgx --title="$title" -- bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    xfce4-terminal)
      xfce4-terminal --title "$title" -e "bash -lc '$cmd; echo; echo \"[$title] finished. Press Enter to close.\"; read'"
      ;;
    mate-terminal)
      mate-terminal --title="$title" -- bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    konsole)
      konsole --new-tab --hold -p tabtitle="$title" -e bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    tilix)
      tilix -t "$title" -e bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    xterm)
      xterm -T "$title" -e bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    alacritty)
      alacritty --title "$title" -e bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    kitty)
      kitty --title "$title" bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    terminator)
      terminator -T "$title" -x bash -lc "$cmd; echo; echo '[$title] finished. Press Enter to close.'; read"
      ;;
    *) return 1 ;;
  esac
}

# Try running a no-op to verify a terminal actually works (DBus issues show up here)
preflight_terminal() {
  local term="$1"
  case "$term" in
    gnome-terminal) gnome-terminal -- bash -lc 'exit' ;;
    kgx)            kgx -- bash -lc 'exit' ;;
    xfce4-terminal) xfce4-terminal -e "bash -lc 'exit'" ;;
    mate-terminal)  mate-terminal -- bash -lc 'exit' ;;
    konsole)        konsole -e bash -lc 'exit' ;;
    tilix)          tilix -e bash -lc 'exit' ;;
    xterm)          xterm -e bash -lc 'exit' ;;
    alacritty)      alacritty -e bash -lc 'exit' ;;
    kitty)          kitty bash -lc 'exit' ;;
    terminator)     terminator -x bash -lc 'exit' ;;
    *) return 1 ;;
  esac
}

detect_terminal() {
  local candidates
  # Allow override: BALDR_TERM=kitty ./run_baldr.sh ...
  if [[ -n "${BALDR_TERM:-}" ]]; then
    candidates=("$BALDR_TERM")
  else
    # Prefer simple X11 terminals first to avoid GNOME/DBus pitfalls
    candidates=(xterm kitty alacritty tilix terminator xfce4-terminal konsole mate-terminal gnome-terminal kgx)
  fi

  for t in "${candidates[@]}"; do
    if command -v "$t" >/dev/null 2>&1; then
      if preflight_terminal "$t" >/dev/null 2>&1; then
        echo "$t"
        return 0
      else
        echo "Terminal '$t' found but failed preflight (DBus/launch issue). Skipping." >&2
      fi
    fi
  done
  return 1
}

TERM_CMD="$(detect_terminal || true)"
if [[ -z "${TERM_CMD:-}" ]]; then
  echo "Error: no working terminal emulator found."
  echo "Quick fix: install xterm and force it:"
  echo "  sudo apt-get install xterm"
  echo "  BALDR_TERM=xterm $0 \"$MODE\" \"$MASK\" \"${BEAM_INPUT}\""
  exit 1
fi

echo "Using terminal: $TERM_CMD"
echo "Mode: $MODE | Mask: $MASK | Beams: ${BEAMS[*]}"

# ---- Launch beams ----
for beam in "${BEAMS[@]}"; do
  if [[ -z "${PORTS[$beam]:-}" ]]; then
    echo "Warning: beam $beam not valid (must be 1–4). Skipping."
    continue
  fi
  port="${PORTS[$beam]}"
  socket="tcp://*:${port}"
  title="Baldr Beam ${beam}"
  cmd="./baldr --beam ${beam} --mode ${MODE} --mask ${MASK} --socket ${socket}"
  echo "Launching: $title  ->  $cmd"
  launch_with "$TERM_CMD" "$title" "$cmd" &
  sleep 0.15
done

wait || true

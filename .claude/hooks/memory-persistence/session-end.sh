#!/bin/bash
# Stop Hook (Session End) - Persist learnings when session ends
#
# Runs when Claude session ends. Creates/updates session log file
# with timestamp for continuity tracking.

set -euo pipefail

sanitize_cell() {
  printf '%s' "$1" | tr '\n' ' ' | tr '|' '-'
}

resolve_session_log() {
  local session_id="$1"
  local worktree="$2"
  local projects_dir="$3"
  local direct_match
  local fallback_dir

  if [ -n "$session_id" ] && [ "$session_id" != "null" ]; then
    direct_match="$(rg --files "$projects_dir" -g "*${session_id}.jsonl" 2>/dev/null | head -n 1 || true)"
    if [ -n "$direct_match" ]; then
      printf '%s' "$direct_match"
      return 0
    fi
  fi

  fallback_dir="$projects_dir/$(printf '%s' "$worktree" | sed 's#^/##; s#/#-#g')"
  if [ -n "$session_id" ] && [ -f "$fallback_dir/$session_id.jsonl" ]; then
    printf '%s' "$fallback_dir/$session_id.jsonl"
    return 0
  fi

  if [ -n "$session_id" ] && [ -f "$fallback_dir/subagents/$session_id.jsonl" ]; then
    printf '%s' "$fallback_dir/subagents/$session_id.jsonl"
    return 0
  fi

  return 1
}

resolve_latest_session_id_for_cwd() {
  local worktree="$1"
  local file
  local sid
  local sid_cwd
  local started_at
  local best_session_id=""
  local best_started_at=0

  for file in "$HOME/.claude/sessions"/*.json; do
    [ -f "$file" ] || continue

    sid="$(jq -r '.sessionId // empty' "$file" 2>/dev/null || true)"
    sid_cwd="$(jq -r '.cwd // empty' "$file" 2>/dev/null || true)"

    [ -n "$sid" ] || continue
    [ "$sid_cwd" = "$worktree" ] || continue

    started_at="$(jq -r '.startedAt // 0' "$file" 2>/dev/null || echo 0)"
    case "$started_at" in
      ''|*[!0-9]*) started_at=0 ;;
    esac

    if [ "$started_at" -gt "$best_started_at" ]; then
      best_started_at=$started_at
      best_session_id=$sid
    fi
  done

  printf '%s' "$best_session_id"
}

sum_session_tokens() {
  local session_log="$1"
  if [ ! -f "$session_log" ]; then
    printf '0'
    return 0
  fi

  jq -r 'select(.type=="assistant" and .message.usage.input_tokens and .message.usage.output_tokens)
    | ((.message.usage.input_tokens // 0) + (.message.usage.output_tokens // 0))
    | tostring' "$session_log" \
    | awk '{ total += $1 } END { printf "%d", (total+0) }'
}

append_memory_row() {
  local common_dir
  local wolf_root
  local memory_file

  common_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
  if [ -z "$common_dir" ]; then
    return 0
  fi

  wolf_root="$(printf '%s' "$common_dir" | sed 's#/.git$##')"
  memory_file="$wolf_root/.wolf/memory.md"

  if [ ! -f "$memory_file" ]; then
    return 0
  fi

  local timestamp
  local worktree
  local status_count
  local row
  local session_id
  local session_log
  local token_count
  local hook_payload
  local projects_dir

  timestamp=$(date '+%H:%M')
  worktree=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
  status_count=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

  hook_payload="$(cat)"
  session_id="$(printf '%s' "$hook_payload" | jq -r '.session_id // .sessionId // empty' 2>/dev/null || true)"
  if [ -z "$session_id" ]; then
    session_id="$(resolve_latest_session_id_for_cwd "$worktree")"
  fi

  projects_dir="$HOME/.claude/projects"
  session_log="$(resolve_session_log "$session_id" "$worktree" "$projects_dir" || true)"
  token_count="$(sum_session_tokens "$session_log")"

  status_count=${status_count:-0}
  token_count=${token_count:-0}

  row="| $timestamp | Session end summary | $(sanitize_cell "$worktree") | $status_count changed/uncommitted files at session end | $token_count |"
  printf '%s\n' "$row" >> "$memory_file"
}

SESSIONS_DIR="${HOME}/.claude/sessions"
TODAY=$(date '+%Y-%m-%d')
SESSION_FILE="${SESSIONS_DIR}/${TODAY}-session.tmp"

mkdir -p "$SESSIONS_DIR"

# If session file exists for today, update the end time
if [ -f "$SESSION_FILE" ]; then
  # Update Last Updated timestamp
  sed -i '' "s/\*\*Last Updated:\*\*.*/\*\*Last Updated:\*\* $(date '+%H:%M')/" "$SESSION_FILE" 2>/dev/null || \
  sed -i "s/\*\*Last Updated:\*\*.*/\*\*Last Updated:\*\* $(date '+%H:%M')/" "$SESSION_FILE" 2>/dev/null
  echo "[SessionEnd] Updated session file: $SESSION_FILE" >&2
else
  # Create new session file with template
  cat > "$SESSION_FILE" << 'EOF_SESSION'
# Session: $(date '+%Y-%m-%d')
**Date:** $TODAY
**Started:** $(date '+%H:%M')
**Last Updated:** $(date '+%H:%M')

---

## Current State

[Session context goes here]

### Completed
- [ ]

### In Progress
- [ ]

### Notes for Next Session
-

### Context to Load
```
[relevant files]
```
EOF_SESSION
  echo "[SessionEnd] Created session file: $SESSION_FILE" >&2
fi

append_memory_row

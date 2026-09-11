#!/bin/bash
# SessionStart Hook - Load previous context on new session
#
# Runs when a new Claude session starts. Checks for recent session
# files and notifies Claude of available context to load.
#
# Hook config (in ~/.claude/settings.json):
# {
#   "hooks": {
#     "SessionStart": [{
#       "matcher": "*",
#       "hooks": [{
#         "type": "command",
#         "command": "~/.claude/hooks/memory-persistence/session-start.sh"
#       }]
#     }]
#   }
# }

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
  local session_file_count
  local learned_count
  local status_count
  local row
  local session_id
  local session_log
  local token_count
  local hook_payload
  local projects_dir

  timestamp=$(date '+%H:%M')
  worktree=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
  session_file_count=$(find "$HOME/.claude/sessions" -name '*.tmp' 2>/dev/null | wc -l | tr -d ' ')
  learned_count=$(find "$HOME/.claude/skills/learned" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
  status_count=$(git status --short 2>/dev/null | wc -l | tr -d ' ')

  hook_payload="$(cat)"
  session_id="$(printf '%s' "$hook_payload" | jq -r '.session_id // .sessionId // empty' 2>/dev/null || true)"
  if [ -z "$session_id" ]; then
    session_id="$(resolve_latest_session_id_for_cwd "$worktree")"
  fi

  projects_dir="$HOME/.claude/projects"
  session_log="$(resolve_session_log "$session_id" "$worktree" "$projects_dir" || true)"
  token_count="$(sum_session_tokens "$session_log")"

  session_file_count=${session_file_count:-0}
  learned_count=${learned_count:-0}
  status_count=${status_count:-0}
  token_count=${token_count:-0}

  row="| $timestamp | Session start context sync | $(sanitize_cell "$worktree") | $status_count staged/uncommitted files, $session_file_count recent session tmp files, $learned_count learned skill notes | $token_count |"
  printf '%s\n' "$row" >> "$memory_file"
}

SESSIONS_DIR="${HOME}/.claude/sessions"
LEARNED_DIR="${HOME}/.claude/skills/learned"

# Check for recent session files (last 7 days)
recent_sessions=$(find "$SESSIONS_DIR" -name "*.tmp" -mtime -7 2>/dev/null | wc -l | tr -d ' ')

if [ "$recent_sessions" -gt 0 ]; then
  latest=$(ls -t "$SESSIONS_DIR"/*.tmp 2>/dev/null | head -1)
  echo "[SessionStart] Found $recent_sessions recent session(s)" >&2
  echo "[SessionStart] Latest: $latest" >&2
fi

# Check for learned skills
learned_count=$(find "$LEARNED_DIR" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')

if [ "$learned_count" -gt 0 ]; then
  echo "[SessionStart] $learned_count learned skill(s) available in $LEARNED_DIR" >&2
fi

# Check for CLAUDE.md staleness carry-over from previous session
STALENESS_LOG="$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/DevSkyy")/.claude/claude-md-staleness.log"
if [ -f "$STALENESS_LOG" ]; then
  # Delete if older than 3 days
  if find "$STALENESS_LOG" -mtime +3 -print -quit 2>/dev/null | grep -q .; then
    rm -f "$STALENESS_LOG"
  else
    echo "[SessionStart] CLAUDE.md staleness detected in previous session:" >&2
    grep -v '^#' "$STALENESS_LOG" | head -10 >&2
    echo "[SessionStart] Run /revise-claude-md to update stale files" >&2
  fi
fi

append_memory_row

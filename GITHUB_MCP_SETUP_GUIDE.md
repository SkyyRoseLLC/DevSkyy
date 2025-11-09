# GitHub MCP Server Setup Guide
**DevSkyy Enterprise Platform**  
**Last Updated:** 2025-11-08

---

## Overview

There are **two different GitHub MCP options** available:

1. **GitHub Copilot MCP** (HTTP) - Requires GitHub Copilot subscription
2. **Standard GitHub MCP** (stdio) - Uses GitHub Personal Access Token

---

## Option 1: GitHub Copilot MCP (HTTP Transport)

### Requirements
- ✅ Active GitHub Copilot subscription ($10/month or $100/year)
- ✅ GitHub Copilot API access
- ✅ Valid Bearer token

### Setup Steps

#### 1. Verify GitHub Copilot Subscription
```bash
# Check if you have Copilot access
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.github.com/user/copilot/seats
```

#### 2. Get GitHub Copilot Token
GitHub Copilot uses a different authentication mechanism than standard PATs.

**Method A: From GitHub Copilot Extension**
1. Install GitHub Copilot extension in VS Code
2. Sign in with GitHub
3. Extract token from extension storage

**Method B: From GitHub CLI**
```bash
gh auth status
gh auth token
```

#### 3. Add to .env File
```bash
# Add to /Users/coreyfoster/DevSkyy/.env
GITHUB_COPILOT_TOKEN=your_copilot_token_here
```

#### 4. Add MCP Server
```bash
# Using your command format
claude mcp add --transport http github-copilot \
  https://api.githubcopilot.com/mcp \
  -H "Authorization: Bearer $(grep GITHUB_COPILOT_TOKEN .env | cut -d '=' -f2)"
```

### Capabilities (GitHub Copilot MCP)
- Code generation with AI assistance
- Context-aware suggestions
- Multi-file editing
- Code review assistance

---

## Option 2: Standard GitHub MCP (stdio - Recommended)

### Requirements
- ✅ GitHub account (free)
- ✅ GitHub Personal Access Token

### Setup Steps

#### 1. Create GitHub Personal Access Token

**Via GitHub Web Interface:**
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Set token name: `DevSkyy MCP Server`
4. Set expiration: 90 days (or custom)
5. Select scopes:
   - ✅ `repo` (Full repository access)
   - ✅ `workflow` (Update GitHub Actions)
   - ✅ `admin:org` (Organization management - if needed)
   - ✅ `write:discussion` (Discussions access)
6. Click **"Generate token"**
7. **Copy the token immediately** (you won't see it again!)

**Via GitHub CLI:**
```bash
gh auth login
gh auth token
```

#### 2. Add Token to .env
```bash
# Add to /Users/coreyfoster/DevSkyy/.env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_actual_token_here
```

#### 3. Verify .env Entry
```bash
grep GITHUB_PERSONAL_ACCESS_TOKEN .env
```

#### 4. Add GitHub MCP Server

**Option A: Using Claude CLI**
```bash
claude mcp add --transport stdio github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=$(grep GITHUB_PERSONAL_ACCESS_TOKEN .env | cut -d '=' -f2) \
  -- npx -y @modelcontextprotocol/server-github
```

**Option B: Manual .mcp.json Update**
Already configured in `/Users/coreyfoster/.mcp.json`:
```json
{
  "github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
    },
    "description": "GitHub repository management"
  }
}
```

Just replace `YOUR_GITHUB_TOKEN_HERE` with your actual token.

### Capabilities (Standard GitHub MCP)
- Create/manage issues
- Create pull requests
- List repositories
- Search code
- Create/edit files
- Manage repository settings

---

## Current Status

### Configured MCP Servers (from `claude mcp list`)

1. **giga** (HTTP) - ⚠️ Needs authentication
   - URL: https://mcp.gigamind.dev/mcp
   
2. **wordpress** (stdio) - ✓ Connected
   - Command: `npx -y @instawp/mcp-wp`
   
3. **wordpress-files** (stdio) - ✓ Connected
   - Filesystem access to WordPress plugin
   
4. **fetch** (stdio) - ✗ Failed to connect
   - Command: `npx -y @modelcontextprotocol/server-fetch`

### Project .mcp.json Servers

From `/Users/coreyfoster/.mcp.json`:
1. filesystem
2. postgres (Neon)
3. github (needs token)
4. gdrive (needs OAuth)
5. brave-search (needs API key)
6. devskyy-orchestrator
7. skyy-rose-voice-media-video

---

## Recommended Approach

### For DevSkyy Project: Use Standard GitHub MCP (stdio)

**Why:**
- ✅ Free (no Copilot subscription needed)
- ✅ Full GitHub API access
- ✅ Works with existing tokens
- ✅ More control over operations
- ✅ Already configured in .mcp.json

**Setup (3 steps):**

```bash
# 1. Create token at GitHub
open https://github.com/settings/tokens

# 2. Add to .env (replace with your actual token)
echo 'GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here' >> .env

# 3. Update .mcp.json with actual token
# (already in file, just needs token value)
```

### For AI-Assisted Coding: Use GitHub Copilot MCP

**Why:**
- ✅ AI-powered code generation
- ✅ Context-aware suggestions
- ✅ Advanced code completion

**Requirements:**
- GitHub Copilot subscription ($10/month)
- Copilot API access

---

## Testing

### Test Standard GitHub MCP
```bash
# Set environment variable
export GITHUB_PERSONAL_ACCESS_TOKEN="your_token"

# Test connection
npx -y @modelcontextprotocol/server-github
```

Should output: "GitHub MCP Server running on stdio"

### Test GitHub Copilot MCP (if configured)
```bash
curl -H "Authorization: Bearer YOUR_COPILOT_TOKEN" \
  https://api.githubcopilot.com/mcp
```

---

## Security Best Practices

### Token Storage
```bash
# GOOD: Store in .env (git ignored)
echo 'GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx' >> .env

# BAD: Hard-code in .mcp.json
# BAD: Commit token to git
# BAD: Share token publicly
```

### Token Permissions
- Only grant necessary scopes
- Use fine-grained tokens when possible
- Rotate tokens regularly (every 90 days)
- Revoke unused tokens

### Secure Retrieval
```bash
# Use macOS Keychain (optional)
security add-generic-password -a "$USER" \
  -s "github_mcp_token" -w "ghp_your_token"

# Retrieve when needed
GITHUB_PERSONAL_ACCESS_TOKEN=$(security find-generic-password \
  -a "$USER" -s "github_mcp_token" -w)
```

---

## Troubleshooting

### Issue: "Package no longer supported"
**Package:** `@modelcontextprotocol/server-github`  
**Status:** Deprecated but functional  
**Action:** Continue using; monitor for replacement

### Issue: Token not found
```bash
# Check .env
grep GITHUB_PERSONAL_ACCESS_TOKEN .env

# Check if token is valid
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/user
```

### Issue: Permission denied
**Cause:** Token lacks necessary scopes  
**Solution:** Recreate token with required permissions

### Issue: HTTP transport not working
**Cause:** GitHub Copilot API requires active subscription  
**Solution:** Use stdio transport with PAT instead

---

## Command Reference

### Add GitHub MCP (stdio)
```bash
claude mcp add --transport stdio github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=your_token \
  -- npx -y @modelcontextprotocol/server-github
```

### Add GitHub Copilot MCP (HTTP)
```bash
claude mcp add --transport http github-copilot \
  https://api.githubcopilot.com/mcp \
  -H "Authorization: Bearer $COPILOT_TOKEN"
```

### List MCP Servers
```bash
claude mcp list
```

### Get Server Details
```bash
claude mcp get github
```

### Remove Server
```bash
claude mcp remove github
```

---

## Next Steps

1. **Immediate:**
   - Create GitHub Personal Access Token
   - Add to `.env` file
   - Update `.mcp.json` with token
   - Test connection

2. **Optional:**
   - Subscribe to GitHub Copilot ($10/month)
   - Add Copilot MCP server
   - Test AI-assisted coding

3. **Best Practice:**
   - Rotate tokens every 90 days
   - Monitor token usage
   - Keep tokens in `.env` (never commit)

---

## Summary

**Recommended Setup:**
```bash
# 1. Get token from GitHub
https://github.com/settings/tokens

# 2. Add to .env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxx

# 3. Your .mcp.json already configured
# Just replace YOUR_GITHUB_TOKEN_HERE with actual token

# 4. Test
export GITHUB_PERSONAL_ACCESS_TOKEN="your_token"
npx -y @modelcontextprotocol/server-github
```

**Status:** Ready to configure once you have GitHub token

---

**Built for The Skyy Rose Collection**  
**DevSkyy Enterprise Platform v5.0.0**

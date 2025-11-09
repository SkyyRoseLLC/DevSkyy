# GitHub MCP Setup - Next Steps

**Status:** Environment prepared, awaiting GitHub token
**Last Updated:** 2025-11-08

---

## What's Been Done

✅ Added `GITHUB_PERSONAL_ACCESS_TOKEN` to `.env` (line 199)
✅ Added GitHub MCP integration section with comments
✅ Updated `.env.example` with template
✅ Included optional MCP integrations (Google Drive, Brave Search)

---

## Next Steps

### 1. Create GitHub Personal Access Token

Go to: **https://github.com/settings/tokens**

1. Click **"Generate new token (classic)"**
2. Set name: `DevSkyy MCP Server`
3. Set expiration: `90 days` (recommended)
4. **Select scopes:**
   - ✅ `repo` - Full repository access
   - ✅ `workflow` - GitHub Actions workflow access
   - ✅ `admin:org` - Organization management (if using orgs)
   - ✅ `write:discussion` - Discussions access
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)

---

### 2. Add Token to .env File

Open `/Users/coreyfoster/DevSkyy/.env` and update line 199:

**Before:**
```bash
GITHUB_PERSONAL_ACCESS_TOKEN=
```

**After:**
```bash
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_actual_token_here
```

**Using Command Line:**
```bash
# Option A: Use nano editor
nano /Users/coreyfoster/DevSkyy/.env

# Navigate to line 199, add your token, save with Ctrl+O, exit with Ctrl+X

# Option B: Use sed (replace YOUR_TOKEN with actual token)
sed -i '' 's/GITHUB_PERSONAL_ACCESS_TOKEN=$/GITHUB_PERSONAL_ACCESS_TOKEN=ghp_YOUR_TOKEN/' /Users/coreyfoster/DevSkyy/.env
```

---

### 3. Verify Token in .env

```bash
cd /Users/coreyfoster/DevSkyy
grep GITHUB_PERSONAL_ACCESS_TOKEN .env
```

Expected output:
```
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 4. Update .mcp.json

The `.mcp.json` file at `/Users/coreyfoster/.mcp.json` needs the token.

**Current state:**
```json
{
  "github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
    }
  }
}
```

**Two options to update:**

**Option A: Automatic from .env**
```bash
# Read token from .env and update .mcp.json
TOKEN=$(grep GITHUB_PERSONAL_ACCESS_TOKEN /Users/coreyfoster/DevSkyy/.env | cut -d '=' -f2)
echo "Token found: ${TOKEN:0:10}..." # Preview first 10 chars

# Update .mcp.json (requires jq)
# Manual update recommended (see Option B)
```

**Option B: Manual update (RECOMMENDED)**
```bash
nano /Users/coreyfoster/.mcp.json

# Replace "YOUR_GITHUB_TOKEN_HERE" with your actual token
# Save and exit
```

---

### 5. Test GitHub MCP Connection

```bash
# Export the token
export GITHUB_PERSONAL_ACCESS_TOKEN=$(grep GITHUB_PERSONAL_ACCESS_TOKEN /Users/coreyfoster/DevSkyy/.env | cut -d '=' -f2)

# Test the server
npx -y @modelcontextprotocol/server-github
```

**Expected output:**
```
GitHub MCP Server running on stdio
```

(Press Ctrl+C to stop)

---

### 6. Verify with Claude CLI

```bash
# List MCP servers
claude mcp list

# Get GitHub server details
claude mcp get github
```

**Expected:** GitHub MCP server should show as connected

---

## Optional: Add Other MCP Integrations

### Google Drive MCP Server

1. **Create Google Cloud Project:**
   - Go to: https://console.cloud.google.com
   - Create new project
   - Enable Google Drive API

2. **Create OAuth2 Credentials:**
   - APIs & Services > Credentials
   - Create OAuth 2.0 Client ID
   - Type: Desktop app
   - Download credentials

3. **Add to .env:**
   ```bash
   GDRIVE_CLIENT_ID=your_client_id.apps.googleusercontent.com
   GDRIVE_CLIENT_SECRET=your_client_secret
   GDRIVE_REFRESH_TOKEN=your_refresh_token
   ```

### Brave Search MCP Server

1. **Get API Key:**
   - Go to: https://brave.com/search/api/
   - Sign up (free tier: 2,000 queries/month)
   - Get API key

2. **Add to .env:**
   ```bash
   BRAVE_API_KEY=BSA_your_api_key_here
   ```

---

## Troubleshooting

### Issue: Token not working

**Check token validity:**
```bash
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

Should return your GitHub user info.

### Issue: .mcp.json not updating

**Verify file location:**
```bash
ls -la /Users/coreyfoster/.mcp.json
cat /Users/coreyfoster/.mcp.json | grep -A5 github
```

### Issue: Server won't start

**Check dependencies:**
```bash
node -v  # Should show v18+
npm -v   # Should show v9+
```

**Clear npm cache:**
```bash
npm cache clean --force
```

---

## Summary

**Current Status:**
- ✅ Environment files prepared
- ⏳ Waiting for GitHub token creation
- ⏳ Token needs to be added to `.env`
- ⏳ `.mcp.json` needs token update

**Estimated Time:** 5-10 minutes

**References:**
- Full guide: `/Users/coreyfoster/DevSkyy/GITHUB_MCP_SETUP_GUIDE.md`
- MCP status: `/Users/coreyfoster/DevSkyy/MCP_SERVER_STATUS_AND_SETUP.md`

---

**Built for The Skyy Rose Collection**
**DevSkyy Enterprise Platform v5.0.0**

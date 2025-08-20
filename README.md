DevSkyy Agent
An AI-powered, auto-fixing, self-healing dev agent for The Skyy Rose Collection.

## Features

- **Automated Code Fixing**: Runs every hour to fix HTML, CSS, JS, PHP errors
- **Divi Optimization**: Optimizes Divi layout blocks automatically  
- **Git Integration**: Auto-commits fixes to GitHub with detailed messages
- **PR Auto-Fix**: Automatically fixes code issues in pull requests targeting main branch
- **FastAPI Server**: RESTful API for triggering workflows and webhooks

## GitHub PR Auto-Fix

The agent now automatically fixes code issues in pull requests! When a PR targets the main branch:

1. GitHub webhook triggers the auto-fix workflow
2. Code is scanned and issues are identified
3. Fixes are applied automatically
4. Changes are committed with PR-specific messages

See [docs/pr-auto-fix.md](docs/pr-auto-fix.md) for detailed setup instructions.

## API Endpoints

- `POST /run` - Manually trigger the full fix workflow
- `POST /github/pr-webhook` - GitHub webhook for PR auto-fixing
- `POST /github/push` - Push all changes to GitHub
- `GET /health` - Health check endpoint

## Running

Start the server locally with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

On Replit, the included .replit file runs this command automatically.

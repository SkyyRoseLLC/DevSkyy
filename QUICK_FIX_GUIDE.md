# DevSkyy Quick Fix Guide

**Date:** 2025-11-04
**Goal:** Get DevSkyy running in 1-2 hours

---

## ✅ Already Fixed (During Review)

1. **tests/conftest.py:11** - Indentation error ✅
2. **models_sqlalchemy.py:6-16** - Malformed try/except ✅
3. **agent/ml_models/forecasting_engine.py:367** - Unclosed parenthesis ✅
4. **agent/modules/backend/ecommerce_agent.py:1-2** - Duplicate imports ✅
5. **database.py:4** - Indented import ✅

---

## 🔧 Quick Fix: Automated Formatting

The fastest way to fix the remaining 30+ syntax errors:

```bash
# Navigate to project
cd /Users/coreyfoster/DevSkyy

# Install formatters (if not installed)
pip install black autopep8

# Run automated formatting
black . --line-length 100 --exclude "venv|htmlcov|skyy-rose"
autopep8 --in-place --aggressive --aggressive --recursive --exclude="venv,htmlcov,skyy-rose" .

# Verify syntax
python3 -m compileall . -q
```

---

## 🚀 Development Environment Setup

### 1. Create .env File (if missing)

```bash
cat > .env << 'EOF'
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Security (generate new keys for production)
SECRET_KEY=dev-secret-key-change-in-production-min-32-chars
JWT_SECRET_KEY=dev-jwt-secret-key-change-in-production-32-chars
JWT_REFRESH_SECRET_KEY=dev-jwt-refresh-key-change-in-production-32

# Database (SQLite for development)
DATABASE_URL=sqlite+aiosqlite:///./devskyy.db

# AI APIs (add your keys)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: Redis (has in-memory fallback)
REDIS_URL=redis://localhost:6379
EOF
```

### 2. Update Dependencies

```bash
# Fix security vulnerability in torch
pip install --upgrade torch==2.6.0 torchvision==0.19.0

# Verify installation
pip list | grep -E "torch|fastapi|anthropic"
```

### 3. Initialize Database

```bash
# Create SQLite database
python3 init_database.py

# Verify database created
ls -lh devskyy.db
```

### 4. Test Application Startup

```bash
# Test imports
python3 -c "import main; print('✅ Imports OK')"

# Start server (Ctrl+C to stop)
python3 main.py

# Should see: "Application startup complete"
```

---

## 🧪 Run Tests

```bash
# Collect tests
pytest tests/ --collect-only

# Run security tests
pytest tests/security/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📝 Manual Fixes (If Automated Formatting Doesn't Work)

### Priority 1: Mismatched Brackets (4 files)

**agent/modules/backend/ecommerce_agent.py:503**
```python
# Line 495: Find opening { and line 503: Change ) to }
```

**agent/ecommerce/order_automation.py:292**
```python
# Line 286: Find opening { and line 292: Change ) to }
```

**security/compliance_monitor.py:485**
```python
# Line 477: Find opening { and line 485: Change ) to }
```

**agent/modules/frontend/design_automation_agent.py:307**
```python
# Check for mismatched brackets around line 307
```

### Priority 2: Critical Indentation Errors

Run this to find all indented first lines:
```bash
grep -n "^    import\|^    from" *.py */*.py */*/*.py 2>/dev/null | head -20
```

Fix by removing leading spaces from first import in each file.

---

## 🎯 Validation Checklist

- [ ] All Python files compile without SyntaxError
- [ ] `python3 main.py` starts without errors
- [ ] Tests can be collected: `pytest --collect-only`
- [ ] JWT auth module loads: `python3 -c "from security.jwt_auth import UserRole"`
- [ ] Database models load: `python3 -c "import models_sqlalchemy"`
- [ ] Torch version ≥ 2.6.0: `pip show torch | grep Version`

---

## 🔒 Security Setup (Production)

### Generate Secure Keys

```bash
# JWT Secret Key (32+ characters)
python3 -c "import secrets; print(f'JWT_SECRET_KEY={secrets.token_urlsafe(32)}')"

# JWT Refresh Secret Key
python3 -c "import secrets; print(f'JWT_REFRESH_SECRET_KEY={secrets.token_urlsafe(32)}')"

# Encryption Master Key
python3 -c "import secrets; print(f'ENCRYPTION_MASTER_KEY={secrets.token_urlsafe(32)}')"
```

Add these to your production .env file (NEVER commit to git).

### SSL/TLS Setup (Production)

```bash
# Option 1: Let's Encrypt (free)
certbot --nginx -d yourdomain.com

# Option 2: Self-signed (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

---

## 📊 Expected Results After Fixes

### Before Fixes
```bash
$ python3 main.py
  File "agent/modules/backend/ecommerce_agent.py", line 503
    ),
    ^
SyntaxError: closing parenthesis ')' does not match opening parenthesis '{' on line 495
```

### After Fixes
```bash
$ python3 main.py
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 🆘 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"
```bash
# Solution: Install missing dependency
pip install -r requirements.txt
```

### Issue: "ImportError: cannot import name 'X' from 'Y'"
```bash
# Solution: Check if file has syntax errors
python3 -m py_compile path/to/file.py
```

### Issue: Database errors
```bash
# Solution: Recreate database
rm devskyy.db
python3 init_database.py
```

### Issue: Redis connection errors
```bash
# Solution: Redis is optional, falls back to in-memory cache
# Or install Redis: brew install redis (Mac) / apt install redis (Linux)
```

---

## 📞 Need Help?

1. Check **CODE_REVIEW_REPORT.md** for detailed analysis
2. Review **AUDIT_REPORT.md** for security compliance status
3. See **IMPLEMENTATION_GUIDE.md** for architecture details
4. Read **CLAUDE.md** for coding standards (Truth Protocol)

---

**Quick Fix Time Estimate: 1-2 hours**
**Full Production Ready: 10-12 hours**

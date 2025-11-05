# API Keys Setup Guide - DevSkyy Platform

This guide provides step-by-step instructions for obtaining all required API keys for the DevSkyy multi-agent platform.

---

## Required API Keys

### 1. **Anthropic Claude API** ✅ (You have this)

**Current Status:** You mentioned you have a valid Claude API key

**How to Get/Verify:**
1. Go to: https://console.anthropic.com/
2. Sign in or create account
3. Navigate to "API Keys" section
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

**Add to .env:**
```bash
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
```

**Pricing:** Pay-as-you-go
- Claude 3.5 Sonnet: $3/$15 per million tokens (input/output)
- Claude 3 Opus: $15/$75 per million tokens

---

### 2. **HuggingFace API** ⚠️ (NEED THIS)

**What it's used for:**
- ML model hosting and inference
- Fashion computer vision models
- Custom model training
- Pre-trained models for sentiment analysis, NLP

**How to Get:**
1. Go to: https://huggingface.co/
2. Click "Sign Up" (top right) or sign in
3. After login, click your profile picture → "Settings"
4. Click "Access Tokens" in left sidebar
5. Click "New token"
   - Name: `DevSkyy Production`
   - Role: Select "Write" (allows model uploads)
6. Click "Generate token"
7. **COPY IMMEDIATELY** (shown only once)

**Add to .env:**
```bash
HUGGINGFACE_API_KEY=hf_YOUR_TOKEN_HERE
# Also add for specific use:
HF_TOKEN=hf_YOUR_TOKEN_HERE
```

**Pricing:** FREE tier available
- Free: Rate-limited inference
- PRO ($9/month): Higher rate limits
- Enterprise: Custom pricing

**Models You'll Access:**
- Fashion classification models
- Image generation (Stable Diffusion)
- Text generation (BERT, GPT variants)
- Style transfer models

---

### 3. **Google Gemini API** ⚠️ (NEED THIS)

**What it's used for:**
- Multi-modal AI (text, images, video)
- Fashion trend analysis
- Advanced reasoning
- Code generation

**How to Get:**
1. Go to: https://makersuite.google.com/app/apikey
   - OR: https://ai.google.dev/
2. Click "Get API Key" or "Create API Key"
3. You may need to:
   - Sign in with Google account
   - Enable Google AI Studio
   - Accept terms of service
4. Click "Create API key in new project" or select existing project
5. Copy the API key

**Add to .env:**
```bash
GOOGLE_API_KEY=AIzaSyYOUR_GEMINI_KEY_HERE
GEMINI_API_KEY=AIzaSyYOUR_GEMINI_KEY_HERE
```

**Pricing:**
- Gemini Pro: FREE up to 60 requests/minute
- Gemini Pro Vision: FREE up to 60 requests/minute
- Gemini Ultra (when available): Paid tier

**Rate Limits (Free):**
- 60 requests per minute
- 1,500 requests per day

---

### 4. **Cursor API** ⚠️ (NEED THIS - IF AVAILABLE)

**What it's used for:**
- AI-powered code generation
- Code completion
- Refactoring assistance

**Status:** Cursor doesn't have a public API yet (as of 2024)

**Alternatives:**
Since Cursor is a desktop IDE and doesn't expose a public API, we'll use:

**Option A: OpenAI Codex API**
1. Go to: https://platform.openai.com/
2. Sign in / Sign up
3. Click "API Keys" (left sidebar)
4. Click "Create new secret key"
5. Name it "DevSkyy Codex"
6. Copy key (starts with `sk-`)

```bash
OPENAI_API_KEY=sk-YOUR_KEY_HERE
OPENAI_CODEX_MODEL=gpt-4  # or gpt-3.5-turbo
```

**Option B: GitHub Copilot API** (if you have access)
1. Go to: https://github.com/settings/copilot
2. Requires GitHub Copilot subscription ($10/month)
3. API access limited to enterprise

**RECOMMENDATION:** Use OpenAI API (you already have this set up) with `gpt-4` model for code generation

---

### 5. **OpenAI API** ✅ (You have this in .env)

**Current in .env:**
```bash
OPENAI_API_KEY=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
```

**Verify it works:**
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**If you need a new one:**
1. https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Name it appropriately
4. Copy immediately

---

## Additional ML/AI Service Keys (Optional but Recommended)

### 6. **Stability AI (Stable Diffusion)** - For Image Generation

**How to Get:**
1. Go to: https://platform.stability.ai/
2. Sign up / Sign in
3. Navigate to API Keys
4. Generate new key

```bash
STABILITY_API_KEY=sk-YOUR_STABILITY_KEY
```

**Pricing:** Credits-based
- $10 = 1,000 credits
- Basic image generation: 1-5 credits per image

---

### 7. **Replicate API** - For Advanced ML Models

**What it's used for:**
- Fashion AI models
- Style transfer
- Advanced image processing

**How to Get:**
1. Go to: https://replicate.com/
2. Sign up
3. Go to: https://replicate.com/account/api-tokens
4. Create token

```bash
REPLICATE_API_TOKEN=r8_YOUR_TOKEN_HERE
```

**Pricing:** Pay per second of GPU time
- Usually $0.0001 - $0.001 per second

---

### 8. **Pinecone** - For Vector Database (ML Embeddings)

**What it's used for:**
- Product recommendations
- Semantic search
- Customer segmentation

**How to Get:**
1. Go to: https://www.pinecone.io/
2. Sign up
3. Create a project
4. Get API key from dashboard

```bash
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_ENVIRONMENT=us-east-1-aws  # or your region
```

**Pricing:**
- Starter: FREE (1 index, 1M vectors)
- Standard: $70/month

---

### 9. **ElevenLabs** - For Voice/Audio Content

**What it's used for:**
- Text-to-speech for voice agents
- Audio content generation

**How to Get:**
1. Go to: https://elevenlabs.io/
2. Sign up
3. Profile → API Keys
4. Generate key

```bash
ELEVENLABS_API_KEY=YOUR_KEY_HERE
```

**Pricing:**
- Free: 10,000 characters/month
- Starter: $5/month (30,000 chars)

---

## Environment Variables Summary

After getting all keys, your `.env` should look like:

```bash
# Core DevSkyy
DEVSKYY_API_KEY=sk_live_ePVW3qg1aspgktnYhPW55ZDiOXRveR5J2H-ucZq7G0k
DEVSKYY_API_URL=http://localhost:8000

# AI Services (REQUIRED)
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
OPENAI_API_KEY=sk-YOUR_KEY_HERE
HUGGINGFACE_API_KEY=hf_YOUR_KEY_HERE
HF_TOKEN=hf_YOUR_KEY_HERE
GOOGLE_API_KEY=AIzaSyYOUR_KEY_HERE
GEMINI_API_KEY=AIzaSyYOUR_KEY_HERE

# ML Services (OPTIONAL)
STABILITY_API_KEY=sk-YOUR_KEY_HERE
REPLICATE_API_TOKEN=r8_YOUR_KEY_HERE
PINECONE_API_KEY=YOUR_KEY_HERE
PINECONE_ENVIRONMENT=us-east-1-aws
ELEVENLABS_API_KEY=YOUR_KEY_HERE

# WordPress
SKYY_ROSE_SITE_URL=https://skyyrose.co
SKYY_ROSE_USERNAME=skyyroseco
SKYY_ROSE_PASSWORD=_LoveHurts107_
SKYY_ROSE_APP_PASSWORD=mrtg sDuG MG2o TH5y 3aMq skUO

# Infrastructure
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://localhost/devskyy
SECRET_KEY=devskyy-secret-key-for-skyy-rose-collection-2024

# Monitoring (OPTIONAL)
SENTRY_DSN=YOUR_SENTRY_DSN
DATADOG_API_KEY=YOUR_DATADOG_KEY
```

---

## Quick Setup Checklist

- [ ] **Anthropic Claude** - https://console.anthropic.com/
- [ ] **HuggingFace** - https://huggingface.co/settings/tokens
- [ ] **Google Gemini** - https://makersuite.google.com/app/apikey
- [ ] **OpenAI** (already have) - Verify it works
- [ ] **Stability AI** - https://platform.stability.ai/ (optional)
- [ ] **Replicate** - https://replicate.com/account/api-tokens (optional)
- [ ] **Pinecone** - https://www.pinecone.io/ (optional)
- [ ] **ElevenLabs** - https://elevenlabs.io/ (optional)

---

## Testing Your Keys

After adding keys to `.env`, test them:

```bash
# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-sonnet-20240229","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'

# Test HuggingFace
curl https://huggingface.co/api/whoami-v2 \
  -H "Authorization: Bearer $HUGGINGFACE_API_KEY"

# Test Gemini
curl "https://generativelanguage.googleapis.com/v1/models?key=$GOOGLE_API_KEY"
```

---

## Cost Estimation

**Minimum Monthly Cost (with free tiers):**
- OpenAI: ~$20-50/month (depending on usage)
- Anthropic Claude: ~$30-100/month
- HuggingFace: FREE (free tier sufficient for development)
- Google Gemini: FREE (60 req/min sufficient)
- **Total Minimum:** ~$50-150/month

**Recommended Monthly Budget:**
- Development: $100-200/month
- Production: $500-1000/month (with scaling)

---

## Security Notes

⚠️ **NEVER commit `.env` to git**
⚠️ **Rotate keys every 90 days**
⚠️ **Use separate keys for dev/staging/prod**
⚠️ **Monitor API usage for unexpected spikes**
⚠️ **Set up billing alerts on all platforms**

---

## Next Steps

1. Get the 3 required API keys (HuggingFace, Gemini, verify Claude)
2. Add them to your `.env` file
3. Run the test commands to verify
4. Proceed with MCP server setup

**Questions?** Check each service's documentation or reach out to their support.

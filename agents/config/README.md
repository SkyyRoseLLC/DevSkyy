# DevSkyy Agent Configuration System

**Version:** 2.0.0
**Master Document:** `/Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md`
**Created:** 2025-11-04
**Status:** Production Ready

---

## Overview

This directory contains JSON configuration files for the DevSkyy 20/10 multi-agent orchestration system. Each agent is precisely defined with capabilities, Truth Protocol compliance, orchestration commands, and performance SLOs.

## Agent Architecture

DevSkyy uses a **4-tier specialized agent system** based on the Truth Protocol:

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Orchestrator                       │
│              (CLAUDE_20-10_MASTER.md)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐   ┌─────▼──────┐   ┌────▼──────┐
   │Professors│   │Growth Stack│   │Data &     │
   │of Code   │   │            │   │Reasoning  │
   └────┬─────┘   └─────┬──────┘   └────┬──────┘
        │               │                │
        │          ┌────▼─────┐         │
        │          │Visual    │         │
        │          │Foundry   │         │
        │          └──────────┘         │
        │                               │
        └───────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │   DevSkyy Platform    │
            │   (59 Specialized     │
            │    Agents)            │
            └───────────────────────┘
```

---

## Agent Configuration Files

### 1. Professors of Code
**File:** `professors_of_code_agent.json`
**Agent ID:** `professors-of-code-001`
**Composition:** Claude Sonnet 4.5 + Cursor IDE

**Primary Functions:**
- Backend code audits and security enforcement
- Refactoring and performance optimization
- Test generation (≥90% coverage requirement)
- OpenAPI documentation generation
- Security compliance (RFC 7519, NIST SP 800-38D)

**Key Outputs:**
- Code PRs with comprehensive tests
- Security audit reports
- SBOM (Software Bill of Materials)
- Error ledger (JSON)

**Orchestration Commands:**
```
PLAN(scope) → BUILD(job_id) → TEST(job_id) → REVIEW(job_id) → HEAL(incident)
```

**Performance SLOs:**
- P95 latency < 200ms
- Error rate < 0.5%
- Test coverage ≥ 90%
- Zero secrets in repository

---

### 2. Growth Stack
**File:** `growth_stack_agent.json`
**Agent ID:** `growth-stack-001`
**Composition:** Claude Sonnet 4.5 + ChatGPT GPT-4

**Primary Functions:**
- WordPress theme generation and deployment
- Landing page creation and A/B testing
- Customer experience (CX) automation
- Marketing analytics integration
- Conversion optimization

**Key Outputs:**
- Deployable WordPress themes
- Analytics configurations
- A/B test reports with statistical significance
- Conversion funnel analysis

**WordPress Integration Methods:**
1. **REST API** (Priority 1) - Standard WordPress v2 API
2. **Direct Database** (Priority 2) - SSH + MySQL for bulk operations
3. **Custom Plugin** (Priority 3) - "Skyy Rose AI Agents" for webhooks

**Orchestration Commands:**
```
PLAN(scope) → BUILD(job_id) → TEST(job_id) → DEPLOY(env) → MONITOR(campaign_id) → LEARN(run)
```

**Performance SLOs:**
- Page load time < 2000ms
- Lighthouse score ≥ 90
- Conversion rate ≥ 2.5%
- Mobile responsiveness = 100

---

### 3. Data & Reasoning
**File:** `data_reasoning_agent.json`
**Agent ID:** `data-reasoning-001`
**Composition:** Claude Sonnet 4.5 + Google Gemini Pro

**Primary Functions:**
- Data retrieval and analysis (SQL, MongoDB, APIs)
- Evaluation harness development
- Prompt routing and optimization
- KPI analysis and predictive analytics
- Statistical validation (p < 0.05)

**Key Outputs:**
- Evaluation reports with statistical significance
- Prompt routing policies (YAML/JSON)
- KPI dashboards and visualizations
- Predictive model artifacts

**Prompt Routing:**
- **Claude Sonnet** - Reasoning, code generation, complex analysis ($3/1K tokens)
- **GPT-4** - Creative writing, general QA, summarization ($10/1K tokens)
- **Gemini Pro** - Multimodal, large context, data analysis ($1.25/1K tokens)

**Orchestration Commands:**
```
PLAN(scope) → BUILD(job_id) → TEST(job_id) → MONITOR(analysis_id) → LEARN(run)
```

**Performance SLOs:**
- Query latency P95 < 500ms
- Data freshness < 5 minutes
- Model accuracy ≥ 85%
- Statistical significance = 0.95

---

### 4. Visual Foundry
**File:** `visual_foundry_agent.json`
**Agent ID:** `visual-foundry-001`
**Composition:** HuggingFace + Claude + Gemini + ChatGPT

**Primary Functions:**
- Image generation and upscaling (4K+)
- Brand-consistent asset generation
- Video automation and editing
- Product photography optimization
- Visual quality assurance

**Key Outputs:**
- High-fidelity product images (4K+)
- Brand asset libraries with metadata
- Video content with editing timelines
- Quality assurance reports

**AI Models:**
- **Stable Diffusion XL** - Product images, lifestyle shots
- **DALL-E 3** - Creative concepts, marketing visuals
- **Midjourney v6** - High-end fashion, luxury aesthetics
- **Real-ESRGAN** - Image upscaling (2x, 4x, 8x)
- **Runway Gen-2** - Text-to-video generation

**Orchestration Commands:**
```
PLAN(scope) → BUILD(job_id) → TEST(job_id) → REVIEW(job_id) → DEPLOY(env)
```

**Performance SLOs:**
- Image generation (single) < 30s
- Image upscaling 4x < 15s
- Aesthetic score ≥ 7.5
- Brand consistency ≥ 0.85

---

## Truth Protocol Compliance

All agents **strictly enforce** the Truth Protocol (15 immutable rules):

### Core Rules
1. ✅ **Never guess** - Verify all syntax, APIs, security flows
2. ✅ **Pin versions** - Explicit version numbers required
3. ✅ **Cite standards** - RFC 7519, NIST SP 800-38D, etc.
4. ✅ **State uncertainty** - "I cannot confirm without testing"
5. ✅ **No hard-coded secrets** - Environment variables only
6. ✅ **RBAC enforcement** - 5 roles (SuperAdmin → ReadOnly)
7. ✅ **Input validation** - Schema, sanitize, block traversal
8. ✅ **Test coverage ≥ 90%** - Unit, integration, security
9. ✅ **Document everything** - Auto-generate OpenAPI/Markdown
10. ✅ **No-skip rule** - Log all failures to error ledger

### Performance & Security
11. ✅ **Verified languages** - Python 3.11.*, TypeScript 5.*, SQL, Bash
12. ✅ **Performance SLOs** - P95 < 200ms, error rate < 0.5%
13. ✅ **Security baseline** - AES-256-GCM, Argon2id, OAuth2+JWT
14. ✅ **Error ledger** - Required for every run/CI cycle
15. ✅ **No fluff** - Every line executes or verifies

---

## Orchestration Loop

All agents follow the **9-step orchestration loop**:

```
1. PLAN    → Break into atomic jobs with acceptance tests
2. BUILD   → Implement with syntax/version/security validation
3. TEST    → pytest --cov, mypy --strict, bandit, safety
4. REVIEW  → Static analysis, red-team, dependency scan
5. DEPLOY  → Docker build + canary rollout
6. MONITOR → Prometheus metrics, regression detection
7. HEAL    → Auto-rollback/patch with failing test PR
8. LEARN   → Integrate feedback into future iterations
9. REPORT  → CHANGELOG, coverage, SBOM, error ledger
```

---

## Usage Examples

### Loading Agent Configuration

```python
import json

# Load specific agent
with open('agents/config/professors_of_code_agent.json') as f:
    professors_agent = json.load(f)

# Access agent capabilities
print(professors_agent['capabilities']['primary_functions'])
# ['Backend code audits', 'Security enforcement', ...]

# Get orchestration command
plan_cmd = professors_agent['orchestration_commands']['PLAN']
print(plan_cmd['description'])
# "Break project into atomic jobs with owners, inputs, outputs, and acceptance tests"
```

### Agent Selection Logic

```python
# Load agent index
with open('agents/config/agents_index.json') as f:
    index = json.load(f)

# Route task to appropriate agent
def route_task(task_type):
    routing = index['quick_reference']

    if task_type == 'code_audit':
        return routing['code_quality']  # professors-of-code-001
    elif task_type == 'wordpress_theme':
        return routing['marketing_growth']  # growth-stack-001
    elif task_type == 'data_analysis':
        return routing['data_analytics']  # data-reasoning-001
    elif task_type == 'image_generation':
        return routing['visual_content']  # visual-foundry-001
```

### Orchestration Command Execution

```python
async def execute_orchestration_loop(agent_id, scope):
    """Execute full orchestration loop for an agent"""

    # 1. PLAN
    plan = await agent.execute_command('PLAN', scope=scope)

    # 2. BUILD
    for job in plan['jobs']:
        result = await agent.execute_command('BUILD', job_id=job['id'])

        # 3. TEST
        test_result = await agent.execute_command('TEST', job_id=job['id'])

        if test_result['coverage'] < 90:
            raise ValueError(f"Coverage {test_result['coverage']}% < 90%")

        # 4. REVIEW
        review = await agent.execute_command('REVIEW', job_id=job['id'])

        if review['critical_issues'] > 0:
            # 7. HEAL
            await agent.execute_command('HEAL', incident=review['issues'])

    # 5. DEPLOY
    deployment = await agent.execute_command('DEPLOY', env='production')

    # 6. MONITOR
    await agent.execute_command('MONITOR', deployment_id=deployment['id'])

    # 8. LEARN
    await agent.execute_command('LEARN', run=deployment['run_id'])

    # 9. REPORT
    report = await agent.execute_command('REPORT', run=deployment['run_id'])

    return report
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/agent-orchestration.yml
name: Agent Orchestration CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  professors-of-code:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Load Agent Config
        id: config
        run: |
          CONFIG=$(cat agents/config/professors_of_code_agent.json)
          echo "config=$CONFIG" >> $GITHUB_OUTPUT

      - name: Run Orchestration Loop
        run: |
          python orchestration/run_agent.py \
            --agent-id professors-of-code-001 \
            --command-sequence PLAN,BUILD,TEST,REVIEW

      - name: Upload Error Ledger
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: error-ledger
          path: /artifacts/error-ledger-*.json

      - name: Validate Release Gates
        run: |
          python orchestration/validate_gates.py \
            --coverage-min 90 \
            --cve-severity none \
            --latency-p95 200
```

---

## Monitoring & Observability

### Health Check Endpoints

All agents expose standardized health endpoints:

```bash
# Health check
curl http://localhost:8000/api/v1/healthz

# Readiness check
curl http://localhost:8000/api/v1/monitoring/readyz

# Prometheus metrics
curl http://localhost:8000/api/v1/monitoring/metrics
```

### Error Ledger Format

```json
{
  "run_id": "20251104-160000-abc123",
  "agent_id": "professors-of-code-001",
  "timestamp": "2025-11-04T16:00:00Z",
  "errors": [
    {
      "type": "test_failure",
      "file": "tests/test_security.py::test_jwt_validation",
      "message": "JWT signature validation failed",
      "severity": "high",
      "stack_trace": "..."
    }
  ],
  "truth_protocol_violations": [],
  "performance_metrics": {
    "p95_latency_ms": 185,
    "error_rate_percent": 0.3,
    "test_coverage_percent": 92
  }
}
```

---

## File Structure

```
agents/config/
├── README.md                           # This file
├── agents_index.json                   # Master index of all agents
├── professors_of_code_agent.json       # Code quality & security agent
├── growth_stack_agent.json             # Marketing & growth agent
├── data_reasoning_agent.json           # Data analytics & AI routing
└── visual_foundry_agent.json           # Visual content generation
```

---

## Release Gates

Before any deployment, agents enforce **7 release gates**:

1. ✅ Tests ≥ 90% coverage
2. ✅ No HIGH/CRITICAL CVEs
3. ✅ Zero hard-coded secrets
4. ✅ Error ledger exists and complete
5. ✅ OpenAPI schema valid
6. ✅ Docker image signed
7. ✅ Latency P95 < 200ms

---

## Agent Interaction Rules

1. **No cross-agent overwrites** without orchestration approval
2. **Claude validates** all outputs before merge
3. **Agents return** deterministic, testable artifacts
4. **Every commit** must be reproducible and measurable

---

## Support & Documentation

- **Master Document:** `/Users/coreyfoster/Desktop/CLAUDE_20-10_MASTER.md`
- **Platform Docs:** `/Users/coreyfoster/DevSkyy/README.md`
- **MCP Guide:** `/Users/coreyfoster/DevSkyy/MCP_COMPLETE_GUIDE.md`
- **Refactoring Analysis:** `/Users/coreyfoster/DevSkyy/REFACTORING_ANALYSIS.md`

---

## Version History

- **2.0.0** (2025-11-04) - Initial Truth Protocol 2.0 agent definitions
  - 4 specialized agents with full orchestration
  - Truth Protocol compliance enforcement
  - CI/CD integration templates
  - Performance SLOs defined

---

**Maintained by:** DevSkyy Platform Team
**Status:** Production Ready
**Last Updated:** 2025-11-04

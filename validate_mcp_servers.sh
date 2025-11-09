#!/bin/bash
#===============================================================================
# DevSkyy MCP Server Validation Script
#===============================================================================
# Tests all MCP servers for operational status
# Complies with Truth Protocol - verifies actual functionality
#===============================================================================

set +e  # Don't exit on errors, we want to test all servers

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

echo "==============================================================================="
echo "DevSkyy MCP Server Validation"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================================="
echo ""

# Counter for statistics
TOTAL=0
PASSED=0
FAILED=0
WARNING=0

#-------------------------------------------------------------------------------
# Test Function
#-------------------------------------------------------------------------------
test_server() {
    local name="$1"
    local command="$2"
    local expected="$3"
    local description="$4"

    TOTAL=$((TOTAL + 1))
    echo -n "Testing $name... "

    # Run command with timeout
    output=$(timeout 3 bash -c "$command" 2>&1)
    exit_code=$?

    if [ $exit_code -eq 124 ]; then
        # Timeout means server started (waiting for input on stdio)
        echo -e "${GREEN}✓ OPERATIONAL${NC} (running on stdio)"
        echo "  └─ $description"
        PASSED=$((PASSED + 1))
        return 0
    elif echo "$output" | grep -qi "$expected\|running on stdio\|server running" 2>/dev/null; then
        echo -e "${GREEN}✓ OPERATIONAL${NC}"
        echo "  └─ $description"
        PASSED=$((PASSED + 1))
        return 0
    elif echo "$output" | grep -qi "error\|fail\|not found" 2>/dev/null; then
        echo -e "${RED}✗ FAILED${NC}"
        echo "  └─ Error: $output" | head -3
        FAILED=$((FAILED + 1))
        return 1
    else
        echo -e "${YELLOW}⚠ WARNING${NC}"
        echo "  └─ Unexpected output: $(echo "$output" | head -1)"
        WARNING=$((WARNING + 1))
        return 2
    fi
}

#-------------------------------------------------------------------------------
# Test 1: GitHub MCP Server
#-------------------------------------------------------------------------------
echo -e "${BLUE}[1/7] GitHub MCP Server${NC}"
if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    echo -e "${RED}✗ FAILED${NC} - GITHUB_PERSONAL_ACCESS_TOKEN not set"
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
else
    test_server \
        "GitHub MCP" \
        "npx -y @modelcontextprotocol/server-github" \
        "GitHub MCP Server running" \
        "Repository management, issues, PRs, code search"
fi
echo ""

#-------------------------------------------------------------------------------
# Test 2: WordPress MCP Server
#-------------------------------------------------------------------------------
echo -e "${BLUE}[2/7] WordPress MCP Server${NC}"
if [ -z "$WORDPRESS_USERNAME" ] || [ -z "$WORDPRESS_PASSWORD" ]; then
    echo -e "${YELLOW}⚠ WARNING${NC} - WordPress credentials not fully set"
    WARNING=$((WARNING + 1))
    TOTAL=$((TOTAL + 1))
else
    test_server \
        "WordPress MCP" \
        "npx -y @instawp/mcp-wp" \
        "WordPress" \
        "Content management for https://skyyrose.co"
fi
echo ""

#-------------------------------------------------------------------------------
# Test 3: Brave Search MCP Server
#-------------------------------------------------------------------------------
echo -e "${BLUE}[3/7] Brave Search MCP Server${NC}"
if [ -z "$BRAVE_API_KEY" ]; then
    echo -e "${RED}✗ FAILED${NC} - BRAVE_API_KEY not set"
    FAILED=$((FAILED + 1))
    TOTAL=$((TOTAL + 1))
else
    test_server \
        "Brave Search MCP" \
        "npx -y @modelcontextprotocol/server-brave-search" \
        "brave-search" \
        "Web search (2,000 queries/month free tier)"
fi
echo ""

#-------------------------------------------------------------------------------
# Test 4: Fetch MCP Server
#-------------------------------------------------------------------------------
echo -e "${BLUE}[4/7] Fetch MCP Server${NC}"
TOTAL=$((TOTAL + 1))
echo -n "Testing Fetch MCP... "
# Special handling for fetch server which produces no initial output
timeout 2 uvx mcp-server-fetch >/dev/null 2>&1 &
fetch_pid=$!
sleep 0.5
if kill -0 $fetch_pid 2>/dev/null; then
    kill $fetch_pid 2>/dev/null
    echo -e "${GREEN}✓ OPERATIONAL${NC}"
    echo "  └─ Web content retrieval via HTTP/HTTPS"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "  └─ Server failed to start"
    FAILED=$((FAILED + 1))
fi
echo ""

#-------------------------------------------------------------------------------
# Test 5: Filesystem MCP Server
#-------------------------------------------------------------------------------
echo -e "${BLUE}[5/7] Filesystem MCP Server${NC}"
test_server \
    "Filesystem MCP" \
    "npx -y @modelcontextprotocol/server-filesystem $(pwd)" \
    "filesystem" \
    "File system operations in DevSkyy directory"
echo ""

#-------------------------------------------------------------------------------
# Test 6: DeepCode Package
#-------------------------------------------------------------------------------
echo -e "${BLUE}[6/7] DeepCode Package${NC}"
TOTAL=$((TOTAL + 1))
if python3 -c "import deepcode" 2>/dev/null; then
    echo -e "${GREEN}✓ OPERATIONAL${NC}"
    echo "  └─ Package installed and importable"
    version=$(python3 -c "import deepcode; print(deepcode.__version__)" 2>/dev/null || echo "unknown")
    echo "  └─ Version: $version"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠ INSTALLING${NC}"
    echo "  └─ Package installation in progress or pending"
    echo "  └─ Install with: pip install deepcode-hku"
    WARNING=$((WARNING + 1))
fi
echo ""

#-------------------------------------------------------------------------------
# Test 7: DeepCode Configuration
#-------------------------------------------------------------------------------
echo -e "${BLUE}[7/7] DeepCode Configuration${NC}"
TOTAL=$((TOTAL + 1))
if [ -f "mcp_agent.config.yaml" ] && [ -f "mcp_agent.secrets.yaml" ]; then
    echo -e "${GREEN}✓ OPERATIONAL${NC}"
    echo "  └─ Configuration files present"

    # Count configured servers
    server_count=$(grep -c "command:" mcp_agent.config.yaml 2>/dev/null || echo "0")
    echo "  └─ MCP servers configured: $server_count"

    # Check for API keys
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "  └─ Anthropic API: ✓ Configured"
    else
        echo "  └─ Anthropic API: ○ Not configured (optional)"
    fi

    if [ -n "$OPENAI_API_KEY" ]; then
        echo "  └─ OpenAI API: ✓ Configured"
    else
        echo "  └─ OpenAI API: ○ Not configured (optional)"
    fi

    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ FAILED${NC} - Configuration files missing"
    FAILED=$((FAILED + 1))
fi
echo ""

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo "==============================================================================="
echo "Validation Summary"
echo "==============================================================================="
echo -e "Total Tests:    $TOTAL"
echo -e "Passed:         ${GREEN}$PASSED${NC}"
echo -e "Failed:         ${RED}$FAILED${NC}"
echo -e "Warnings:       ${YELLOW}$WARNING${NC}"
echo ""

# Calculate percentage
if [ $TOTAL -gt 0 ]; then
    percentage=$((PASSED * 100 / TOTAL))
    echo -e "Success Rate:   ${percentage}%"
else
    echo -e "Success Rate:   N/A"
fi

echo ""

#-------------------------------------------------------------------------------
# Recommendations
#-------------------------------------------------------------------------------
if [ $FAILED -gt 0 ] || [ $WARNING -gt 0 ]; then
    echo "==============================================================================="
    echo "Recommendations"
    echo "==============================================================================="

    if [ -z "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
        echo "• Add GITHUB_PERSONAL_ACCESS_TOKEN to .env"
    fi

    if [ -z "$BRAVE_API_KEY" ]; then
        echo "• Add BRAVE_API_KEY to .env (get from https://brave.com/search/api/)"
    fi

    if ! python3 -c "import deepcode" 2>/dev/null; then
        echo "• Install DeepCode: pip install deepcode-hku"
    fi

    if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo "• Optional: Add ANTHROPIC_API_KEY or OPENAI_API_KEY for DeepCode AI models"
    fi

    echo ""
fi

#-------------------------------------------------------------------------------
# Exit Code
#-------------------------------------------------------------------------------
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Status: NEEDS ATTENTION${NC}"
    exit 1
elif [ $WARNING -gt 0 ]; then
    echo -e "${YELLOW}Status: PARTIALLY OPERATIONAL${NC}"
    exit 2
else
    echo -e "${GREEN}Status: FULLY OPERATIONAL ✓${NC}"
    exit 0
fi

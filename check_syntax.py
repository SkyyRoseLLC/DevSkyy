#!/usr/bin/env python3
"""
Syntax checker for DevSkyy codebase
Identifies all Python files with syntax errors
"""

import py_compile
import os
from pathlib import Path

def check_syntax():
    """Check syntax of all Python files"""
    errors = []
    successes = []

    # Find all Python files
    for py_file in Path(".").rglob("*.py"):
        # Skip virtual environments and git
        if any(skip in str(py_file) for skip in [".venv", "venv", ".git", "node_modules"]):
            continue

        try:
            py_compile.compile(str(py_file), doraise=True)
            successes.append(str(py_file))
        except SyntaxError as e:
            errors.append({
                "file": str(py_file),
                "line": e.lineno,
                "msg": e.msg,
                "text": e.text.strip() if e.text else ""
            })
        except Exception as e:
            errors.append({
                "file": str(py_file),
                "line": "?",
                "msg": str(e),
                "text": ""
            })

    # Report results
    print(f"✓ {len(successes)} files OK")
    print(f"✗ {len(errors)} files with errors\n")

    if errors:
        print("=" * 80)
        print("SYNTAX ERRORS FOUND:")
        print("=" * 80)
        for i, err in enumerate(errors[:30], 1):  # Show first 30 errors
            print(f"\n{i}. {err['file']}:{err['line']}")
            print(f"   Error: {err['msg']}")
            if err['text']:
                print(f"   Code: {err['text']}")

    return errors

if __name__ == "__main__":
    errors = check_syntax()
    exit(1 if errors else 0)

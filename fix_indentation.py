#!/usr/bin/env python3
"""
Fix leading whitespace in Python files
Removes unexpected indentation at the start of files
"""

import sys
from pathlib import Path

def fix_leading_whitespace(file_path):
    """Remove leading whitespace from Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find first non-empty line
        first_content_idx = 0
        for i, line in enumerate(lines):
            if line.strip():
                first_content_idx = i
                break

        # Check if first line has unexpected indentation
        if first_content_idx < len(lines):
            first_line = lines[first_content_idx]
            if first_line[0].isspace() and not first_line.strip().startswith('#'):
                # File has leading whitespace
                # Strip leading whitespace from all lines
                fixed_lines = []
                for line in lines:
                    if line.strip():  # Non-empty line
                        # Remove leading whitespace
                        fixed_lines.append(line.lstrip())
                    else:
                        fixed_lines.append(line)

                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)

                return True

        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

# Files to fix
files_to_fix = [
    "create_user.py",
    "deployment_verification.py",
    "update_action_shas.py",
    "init_database.py",
    "startup_sqlalchemy.py",
    "test_vercel_deployment.py",
    "tools/todo_tracker.py",
    "security/enhanced_security.py",
    "tests/test_basic_functionality.py",
    "tests/ml/test_ml_infrastructure.py",
    "fashion/intelligence_engine.py",
    "agent/ecommerce/customer_intelligence.py",
    "agent/wordpress/seo_optimizer.py",
    "agent/wordpress/content_generator.py",
    "agent/modules/enhanced_learning_scheduler.py",
    "agent/modules/frontend/autonomous_landing_page_generator.py",
    "agent/modules/backend/fixer.py",
    "agent/modules/backend/enhanced_brand_intelligence_agent.py",
]

if __name__ == "__main__":
    fixed_count = 0
    for file in files_to_fix:
        file_path = Path(file)
        if file_path.exists():
            if fix_leading_whitespace(file_path):
                print(f"✓ Fixed: {file}")
                fixed_count += 1
            else:
                print(f"- Skipped: {file} (no leading whitespace)")
        else:
            print(f"✗ Not found: {file}")

    print(f"\n{fixed_count} files fixed")

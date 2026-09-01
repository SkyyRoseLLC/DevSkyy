"""Regression coverage for the controlled render-QC provider override."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_openai_qc_provider_is_opt_in_via_scoped_environment() -> None:
    """A caller can explicitly select the documented fallback without changing default QC."""
    env = os.environ.copy()
    env["OAI_QC_JUDGE_PROVIDER"] = "openai"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.oai_render import config; print(config.QC_JUDGE_PROVIDER)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "openai"

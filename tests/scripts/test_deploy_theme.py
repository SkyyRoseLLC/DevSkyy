"""Tests for scripts/deploy-theme.sh -- subprocess-based verification.

Tests invoke the deploy script via subprocess and assert on exit codes,
stdout/stderr content, and script source patterns. Uses temporary directories
with fake .env.wordpress for controlled testing.
"""

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "deploy-theme.sh"


@pytest.fixture
def fake_env(tmp_path):
    """Create a fake .env.wordpress and theme directory for testing."""
    env_file = tmp_path / ".env.wordpress"
    env_file.write_text(
        "SSH_HOST=ssh.wp.com\n"
        "SSH_PORT=22\n"
        "SSH_USER=test.wordpress.com\n"
        "SSH_PASS=fake-password\n"
        "WP_THEME_PATH=/htdocs/wp-content/themes/skyyrose-flagship\n"
        "SFTP_HOST=sftp.wp.com\n"
        "SFTP_PORT=22\n"
        "SFTP_USER=test.wordpress.com\n"
        "SFTP_PASS=fake-password\n"
    )
    theme_dir = tmp_path / "wordpress-theme" / "skyyrose-flagship"
    _make_deployable_theme(theme_dir, version="1.0.0")
    return tmp_path, env_file, theme_dir


@pytest.fixture
def fake_v2_staging_env(tmp_path):
    """Create an explicit staging target for the V2 theme."""
    env_file = tmp_path / ".env.wordpress.staging"
    env_file.write_text(
        "SSH_HOST=ssh.wp.com\n"
        "SSH_PORT=22\n"
        "SSH_USER=staging.wordpress.com\n"
        "SSH_PASS=fake-password\n"
        "WP_THEME_PATH=/htdocs/wp-content/themes/skyyrose-flagship-2\n"
        "SFTP_HOST=sftp.wp.com\n"
        "SFTP_PORT=22\n"
        "SFTP_USER=staging.wordpress.com\n"
        "SFTP_PASS=fake-password\n"
        "PUBLIC_URL=https://staging.skyyrose.co/\n"
    )
    theme_dir = tmp_path / "wordpress-theme" / "skyyrose-flagship-2"
    _make_deployable_theme(theme_dir, version="2.4.4", generation=2)
    return tmp_path, env_file, theme_dir


def _make_deployable_theme(theme_dir: Path, *, version: str = "1.0.0", generation: int = 1) -> None:
    """Write a minimal but *gate-passing* theme.

    preflight_completeness() requires a synced version triple + the
    critical-asset floor (>=3 emblem webp, >=10 woff2, skyy.glb). Transport/
    ordering/cache tests need a theme that clears the gate, not an exhaustive
    one — so this is the smallest tree that deploys. Kept in one helper so the
    floor's magic numbers live in a single place if they ever change.
    """
    theme_dir.mkdir(parents=True, exist_ok=True)
    (theme_dir / "style.css").write_text(
        f"/*\nTheme Name: Test\nVersion:             {version}\n*/\n"
    )
    version_constant = "SKYYROSE2_VERSION" if generation == 2 else "SKYYROSE_VERSION"
    (theme_dir / "functions.php").write_text(
        f"<?php\ndefine( '{version_constant}', '{version}' );\n"
    )
    (theme_dir / "readme.txt").write_text(f"Stable tag: {version}\n")
    if generation == 2:
        for relative_path in (
            "assets/css/design-tokens.min.css",
            "assets/css/theme.min.css",
            "assets/js/theme.min.js",
        ):
            path = theme_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        fonts = theme_dir / "assets" / "sot" / "fonts"
        fonts.mkdir(parents=True, exist_ok=True)
        for i in range(9):
            (fonts / f"font-{i}-latin.woff2").write_bytes(b"")
        models = theme_dir / "assets" / "models"
        models.mkdir(parents=True, exist_ok=True)
        (models / "skyy-mascot.glb").write_bytes(b"")
        return

    emblems = theme_dir / "assets" / "images" / "emblems"
    emblems.mkdir(parents=True, exist_ok=True)
    for name in ("black-rose", "love-hurts", "signature"):
        (emblems / f"{name}-emblem.webp").write_bytes(b"")
    fonts = theme_dir / "assets" / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (fonts / f"font-{i}-latin.woff2").write_bytes(b"")
    models = theme_dir / "assets" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "skyy.glb").write_bytes(b"")


def run_script(*args, env_overrides=None):
    """Run deploy-theme.sh with given arguments."""
    env = os.environ.copy()
    # Isolate from real deploys: the script's concurrency lock and log default
    # to shared /tmp paths, so an in-flight production deploy fails these tests
    # ("Another deploy is already running") and test runs pollute /tmp with
    # skyyrose-deploy-*.log files.
    pid = os.getpid()
    env.setdefault("DEPLOY_LOCK_FILE", f"/tmp/skyyrose-deploy-test-{pid}.lock")
    env.setdefault("DEPLOY_LOG_FILE", f"/tmp/skyyrose-deploy-test-{pid}.log")
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result


class TestDryRun:
    """Test 1: --dry-run exits 0 and prints DRY RUN messages."""

    def test_dry_run_exits_zero(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_dry_run_prints_dry_run_label(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert "[DRY RUN]" in result.stdout

    def test_dry_run_mentions_maintenance_mode(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert "maintenance" in result.stdout.lower()

    def test_dry_run_mentions_file_transfer(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        assert "transfer" in output or "rsync" in output or "lftp" in output

    def test_dry_run_mentions_cache_flush(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert "cache flush" in result.stdout.lower()

    def test_v1_resolves_archive_root_and_remote_path(self, fake_env):
        _, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "Archive root: skyyrose-flagship" in result.stdout
        assert "Remote theme: /htdocs/wp-content/themes/skyyrose-flagship" in result.stdout
        assert "Version triple in sync: 1.0.0" in result.stdout

    def test_v2_staging_resolves_exact_target(self, fake_v2_staging_env):
        _, env_file, theme_dir = fake_v2_staging_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "DEPLOY_TARGET": "staging",
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "Deploy target: staging" in result.stdout
        assert "Archive root: skyyrose-flagship-2" in result.stdout
        assert "Remote theme: /htdocs/wp-content/themes/skyyrose-flagship-2" in result.stdout
        assert "Public URL: https://staging.skyyrose.co/" in result.stdout
        assert "Version triple in sync: 2.4.4" in result.stdout

    def test_rejects_dot_segment_archive_root(self, fake_env):
        _, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir / ".."),
            },
        )
        assert result.returncode != 0
        assert "Unsafe theme archive root '..'" in (result.stdout + result.stderr)

    @pytest.mark.parametrize(
        ("env_updates", "message"),
        [
            ({"REMOTE_DEPLOY_DIR": "/tmp/.."}, "Unsafe remote deploy directory"),
            (
                {"REMOTE_DEPLOY_DIR": "/htdocs/wp-content/themes"},
                "overlaps live theme",
            ),
        ],
    )
    def test_rejects_unsafe_or_overlapping_remote_deploy_dir(self, fake_env, env_updates, message):
        _, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
                **env_updates,
            },
        )
        assert result.returncode != 0
        assert message in (result.stdout + result.stderr)

    @pytest.mark.parametrize(
        ("env_updates", "message"),
        [
            ({"PUBLIC_URL": "https://skyyrose.co/"}, "Staging PUBLIC_URL"),
            (
                {"WP_THEME_PATH": "/htdocs/wp-content/themes/skyyrose-flagship"},
                "does not match archive root",
            ),
        ],
    )
    def test_v2_staging_rejects_production_target(self, fake_v2_staging_env, env_updates, message):
        _, env_file, theme_dir = fake_v2_staging_env
        env_text = env_file.read_text()
        for key, value in env_updates.items():
            env_text = re.sub(rf"^{key}=.*$", f"{key}={value}", env_text, flags=re.MULTILINE)
        env_file.write_text(env_text)
        result = run_script(
            "--dry-run",
            env_overrides={
                "DEPLOY_TARGET": "staging",
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode != 0
        assert message in (result.stdout + result.stderr)

    def test_v2_staging_requires_staging_named_env_file(self, fake_v2_staging_env):
        tmp_path, env_file, theme_dir = fake_v2_staging_env
        generic_env = tmp_path / ".env.wordpress"
        env_file.rename(generic_env)
        result = run_script(
            "--dry-run",
            env_overrides={
                "DEPLOY_TARGET": "staging",
                "ENV_FILE": str(generic_env),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode != 0
        assert "staging-specific ENV_FILE" in (result.stdout + result.stderr)

    def test_v2_staging_requires_public_url(self, fake_v2_staging_env):
        _, env_file, theme_dir = fake_v2_staging_env
        env_file.write_text(
            re.sub(r"^PUBLIC_URL=.*$\n?", "", env_file.read_text(), flags=re.MULTILINE)
        )
        result = run_script(
            "--dry-run",
            env_overrides={
                "DEPLOY_TARGET": "staging",
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
                "PUBLIC_URL": "",
            },
        )
        assert result.returncode != 0
        assert "explicit PUBLIC_URL" in (result.stdout + result.stderr)

    def test_env_file_cannot_downgrade_staging_target(self, fake_v2_staging_env):
        _, env_file, theme_dir = fake_v2_staging_env
        env_file.write_text(env_file.read_text() + "DEPLOY_TARGET=production\n")
        result = run_script(
            "--dry-run",
            env_overrides={
                "DEPLOY_TARGET": "staging",
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        assert result.returncode == 0, result.stderr
        assert "Deploy target: staging" in result.stdout


class TestHelp:
    """Test 2: --help exits 0 and prints usage."""

    def test_help_exits_zero(self):
        result = run_script("--help")
        assert result.returncode == 0

    def test_help_prints_usage(self):
        result = run_script("--help")
        output = result.stdout.lower()
        assert "usage" in output or "deploy" in output


class TestMissingEnv:
    """Test 3: Missing .env.wordpress causes non-zero exit."""

    def test_missing_env_exits_nonzero(self, tmp_path):
        result = run_script(
            env_overrides={"ENV_FILE": str(tmp_path / "nonexistent.env")},
        )
        assert result.returncode != 0

    def test_missing_env_prints_error(self, tmp_path):
        result = run_script(
            env_overrides={"ENV_FILE": str(tmp_path / "nonexistent.env")},
        )
        output = (result.stdout + result.stderr).lower()
        assert "credential" in output or "env" in output or "not found" in output


class TestTrapCleanup:
    """Test 4: Script contains trap cleanup EXIT and cleanup checks MAINTENANCE_ACTIVE."""

    def test_trap_cleanup_exit_registered(self):
        source = SCRIPT_PATH.read_text()
        assert "trap cleanup EXIT" in source or "trap cleanup EXIT INT TERM" in source

    def test_cleanup_checks_maintenance_active(self):
        source = SCRIPT_PATH.read_text()
        assert "MAINTENANCE_ACTIVE" in source
        # cleanup function should check the flag
        cleanup_match = re.search(
            r"cleanup\s*\(\)\s*\{[^}]+MAINTENANCE_ACTIVE[^}]+\}", source, re.DOTALL
        )
        assert cleanup_match is not None, "cleanup() must check MAINTENANCE_ACTIVE"

    def test_trap_before_maintenance_activation(self):
        """Trap must be registered BEFORE any maintenance mode activation."""
        source = SCRIPT_PATH.read_text()
        trap_pos = source.find("trap cleanup")
        activate_pos = source.find("maintenance-mode activate")
        assert trap_pos < activate_pos, "trap must be registered before maintenance-mode activate"


class TestCommandOrdering:
    """Test 5: In dry-run output, activate before transfer, deactivate after."""

    def test_maintenance_activates_before_file_transfer(self, fake_env):
        """Maintenance mode must activate before files are transferred (--with-maintenance path)."""
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            "--with-maintenance",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        activate_pos = output.find("maintenance-mode activate")
        # Look for transfer indicators
        transfer_pos = min(
            (
                p
                for p in (
                    output.find("rsync"),
                    output.find("transfer"),
                    output.find("lftp"),
                    output.find("sftp"),
                    output.find("upload"),
                )
                if p != -1
            ),
            default=-1,
        )
        assert activate_pos != -1, "activate not found in output"
        assert transfer_pos != -1, "transfer not found in output"
        # Deploy script activates maintenance BEFORE transferring files
        assert (
            activate_pos < transfer_pos
        ), "maintenance-mode activate must appear before file transfer"

    def test_deactivate_after_transfer(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            "--with-maintenance",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        transfer_pos = min(
            (
                p
                for p in (
                    output.find("rsync"),
                    output.find("transfer"),
                    output.find("lftp"),
                    output.find("sftp"),
                    output.find("upload"),
                )
                if p != -1
            ),
            default=-1,
        )
        deactivate_pos = output.find("maintenance-mode deactivate")
        assert transfer_pos != -1, "transfer not found"
        assert deactivate_pos != -1, "deactivate not found"
        assert transfer_pos < deactivate_pos


class TestCacheFlush:
    """Test 6: Cache flush commands appear after file transfer."""

    def test_cache_flush_in_output(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        assert "cache flush" in output

    def test_transient_delete_in_output(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        assert "transient delete" in output

    def test_rewrite_flush_in_output(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        assert "rewrite flush" in output

    def test_cache_flush_after_transfer(self, fake_env):
        tmp_path, env_file, theme_dir = fake_env
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
            },
        )
        output = result.stdout.lower()
        transfer_pos = min(
            (
                p
                for p in (
                    output.find("rsync"),
                    output.find("transfer"),
                    output.find("lftp"),
                    output.find("sftp"),
                    output.find("upload"),
                )
                if p != -1
            ),
            default=-1,
        )
        cache_pos = output.find("cache flush")
        assert transfer_pos < cache_pos


def _defs_only_script() -> str:
    """scripts/deploy-theme.sh with the trailing `main "$@"` call stripped, so
    sourcing it only defines functions/arrays and never executes a deploy."""
    lines = SCRIPT_PATH.read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    assert lines[-1].strip() == 'main "$@"', f"expected trailing main call, got: {lines[-1]!r}"
    return "\n".join(lines[:-1]) + "\n"


def _rendered_excludes() -> tuple[list[str], list[str]]:
    """The actual RSYNC_EXCLUDES / tar_excludes arrays as built at runtime --
    not a static grep of the source text, which breaks the moment the exclude
    lists move from literal arrays to any data-driven construction (as they
    did in the 2026-08-02 dedup refactor; see SKYY_EXCLUDE_COMMON_* +
    skyyrose_render_rsync_excludes/skyyrose_render_tar_excludes)."""
    script = (
        _defs_only_script()
        + '\nprintf "RSYNC:%s\\n" "${RSYNC_EXCLUDES[@]}"\n'
        + 'skyyrose_render_tar_excludes | sed "s/^/TAR:/"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=15, check=True
    )
    lines = result.stdout.splitlines()
    rsync_excludes = [line[len("RSYNC:") :] for line in lines if line.startswith("RSYNC:")]
    tar_excludes = [line[len("TAR:") :] for line in lines if line.startswith("TAR:")]
    return rsync_excludes, tar_excludes


def _run_defs_script(body: str) -> subprocess.CompletedProcess[str]:
    """Source deploy function definitions and run a focused shell harness."""
    return subprocess.run(
        ["bash", "-c", _defs_only_script() + "\n" + body],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _render_remote_swap_command(
    *, remote_dir: Path, theme_root: str, theme_path: Path, archive_name: str
) -> str:
    body = f"""
THEME_ARCHIVE_ROOT={theme_root!s}
WP_THEME_PATH={theme_path!s}
REMOTE_DEPLOY_DIR={remote_dir!s}
render_remote_swap_command '' {archive_name!s} test-swap
"""
    result = _run_defs_script(body)
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestRemoteTargetIdentity:
    """Staging credentials must resolve to the canonical staging WordPress."""

    def test_matching_remote_site_url_passes(self):
        result = _run_defs_script("""
DEPLOY_TARGET=staging
DRY_RUN=false
PUBLIC_URL=https://staging.skyyrose.co/
read_remote_site_url() { printf '%s\\n' 'https://staging.skyyrose.co'; }
verify_remote_target_identity
""")
        assert result.returncode == 0, result.stderr
        assert "identity verified" in result.stdout

    def test_production_remote_site_url_fails_closed(self):
        result = _run_defs_script("""
DEPLOY_TARGET=staging
DRY_RUN=false
PUBLIC_URL=https://staging.skyyrose.co/
read_remote_site_url() { printf '%s\\n' 'https://skyyrose.co'; }
if verify_remote_target_identity; then exit 90; fi
""")
        assert result.returncode == 0
        assert "does not match staging target" in result.stderr

    def test_unreadable_remote_site_url_fails_closed(self):
        result = _run_defs_script("""
DEPLOY_TARGET=staging
DRY_RUN=false
PUBLIC_URL=https://staging.skyyrose.co/
read_remote_site_url() { return 1; }
if verify_remote_target_identity; then exit 91; fi
""")
        assert result.returncode == 0
        assert "Unable to read remote WordPress site URL" in result.stderr


class TestRemoteSwapPlan:
    """Execute the rendered remote plan without contacting WordPress."""

    @staticmethod
    def _write_archive(remote_dir: Path, root_name: str) -> str:
        source_root = remote_dir / "archive-source" / root_name
        source_root.mkdir(parents=True)
        (source_root / "candidate.txt").write_text("candidate")
        archive = remote_dir / "candidate.tar"
        with tarfile.open(archive, "w") as handle:
            handle.add(source_root, arcname=root_name)
        return archive.name

    def test_missing_extracted_root_preserves_live_theme(self, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        live_theme = tmp_path / "themes" / "skyyrose-flagship-2"
        live_theme.mkdir(parents=True)
        (live_theme / "live.txt").write_text("live")
        archive_name = self._write_archive(remote_dir, "wrong-root")
        command = _render_remote_swap_command(
            remote_dir=remote_dir,
            theme_root="skyyrose-flagship-2",
            theme_path=live_theme,
            archive_name=archive_name,
        )

        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

        assert result.returncode != 0
        assert (live_theme / "live.txt").read_text() == "live"
        assert not Path(f"{live_theme}.old.test-swap").exists()

    def test_candidate_move_failure_restores_live_theme(self, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        live_theme = tmp_path / "themes" / "skyyrose-flagship-2"
        live_theme.mkdir(parents=True)
        (live_theme / "live.txt").write_text("live")
        archive_name = self._write_archive(remote_dir, "skyyrose-flagship-2")
        command = _render_remote_swap_command(
            remote_dir=remote_dir,
            theme_root="skyyrose-flagship-2",
            theme_path=live_theme,
            archive_name=archive_name,
        )
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_mv = fake_bin / "mv"
        fake_mv.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "skyyrose-flagship-2" ]]; then exit 23; fi\n'
            'exec /bin/mv "$@"\n'
        )
        fake_mv.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"

        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env)

        assert result.returncode == 23
        assert (live_theme / "live.txt").read_text() == "live"
        assert not Path(f"{live_theme}.old.test-swap").exists()

    def test_successful_swap_retains_rollback_generation(self, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        stale_root = remote_dir / "skyyrose-flagship-2"
        stale_root.mkdir()
        (stale_root / "stale.txt").write_text("must not ship")
        live_theme = tmp_path / "themes" / "skyyrose-flagship-2"
        live_theme.mkdir(parents=True)
        (live_theme / "live.txt").write_text("live")
        archive_name = self._write_archive(remote_dir, "skyyrose-flagship-2")
        command = _render_remote_swap_command(
            remote_dir=remote_dir,
            theme_root="skyyrose-flagship-2",
            theme_path=live_theme,
            archive_name=archive_name,
        )

        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert (live_theme / "candidate.txt").read_text() == "candidate"
        assert not (live_theme / "stale.txt").exists()
        backup = Path(f"{live_theme}.old.test-swap")
        assert (backup / "live.txt").read_text() == "live"
        assert not (remote_dir / archive_name).exists()


class TestStagingFallback:
    def test_staging_never_calls_non_atomic_lftp_fallback(self):
        result = _run_defs_script("""
DEPLOY_TARGET=staging
THEME_DIR=/tmp/source
WP_THEME_PATH=/tmp/target
try_rsync() { return 1; }
try_lftp() { echo LFTP_CALLED; return 0; }
if transfer_files; then exit 92; fi
""")
        assert result.returncode == 0
        assert "non-atomic lftp mirror fallback is disabled" in result.stderr
        assert "LFTP_CALLED" not in result.stdout


class TestExcludes:
    """Test 7: Rsync exclude list covers critical patterns."""

    def test_excludes_node_modules(self):
        source = SCRIPT_PATH.read_text()
        assert "node_modules" in source

    def test_excludes_git(self):
        rsync_excludes, tar_excludes = _rendered_excludes()
        assert "--exclude=.git" in rsync_excludes
        assert "--exclude=.git" in tar_excludes

    def test_excludes_env_files(self):
        source = SCRIPT_PATH.read_text()
        assert ".env" in source

    def test_excludes_map_files(self):
        source = SCRIPT_PATH.read_text()
        assert "*.map" in source

    def test_excludes_tests(self):
        rsync_excludes, tar_excludes = _rendered_excludes()
        assert "--exclude=tests/" in rsync_excludes
        assert "--exclude=tests" in tar_excludes

    def test_excludes_package_json(self):
        source = SCRIPT_PATH.read_text()
        assert "package.json" in source

    def test_excludes_package_lock(self):
        source = SCRIPT_PATH.read_text()
        assert "package-lock.json" in source


class TestShellcheck:
    """Test 8: shellcheck passes on the deploy script."""

    @pytest.mark.skipif(
        not shutil.which("shellcheck"),
        reason="shellcheck binary not installed",
    )
    def test_shellcheck_passes(self):
        result = subprocess.run(
            ["shellcheck", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"shellcheck errors:\n{result.stdout}"


class TestCompletenessGate:
    """Test 9: preflight_completeness() fails CLOSED on an incomplete source
    tree (bug-252 gate) and never crashes with a raw awk/sed error (bug-253)."""

    def test_missing_version_file_fails_closed_cleanly(self, fake_env):
        """A missing readme.txt must exit non-zero with the gate's own clear
        message -- not a raw 'awk: can't open file' crash (regression bug-253)."""
        tmp_path, env_file, theme_dir = fake_env
        (theme_dir / "readme.txt").unlink()
        result = run_script(
            "--dry-run",
            env_overrides={"ENV_FILE": str(env_file), "THEME_DIR_OVERRIDE": str(theme_dir)},
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "Version file missing" in combined, f"expected clean gate message, got: {combined}"
        assert (
            "can't open file" not in combined
        ), "raw awk crash leaked -- gate did not fail closed gracefully"

    def test_version_drift_fails_closed(self, fake_env):
        """A synced-but-mismatched version triple must block the deploy."""
        tmp_path, env_file, theme_dir = fake_env
        (theme_dir / "readme.txt").write_text("Stable tag: 9.9.9\n")
        result = run_script(
            "--dry-run",
            env_overrides={"ENV_FILE": str(env_file), "THEME_DIR_OVERRIDE": str(theme_dir)},
        )
        assert result.returncode != 0
        assert "DRIFT" in (result.stdout + result.stderr)

    def test_asset_floor_fails_closed(self, fake_env):
        """Dropping below the critical-asset floor (emblems) must block."""
        tmp_path, env_file, theme_dir = fake_env
        for webp in (theme_dir / "assets" / "images" / "emblems").glob("*.webp"):
            webp.unlink()
        result = run_script(
            "--dry-run",
            env_overrides={"ENV_FILE": str(env_file), "THEME_DIR_OVERRIDE": str(theme_dir)},
        )
        assert result.returncode != 0
        assert "Critical-asset floor" in (result.stdout + result.stderr)

    def test_skip_env_bypasses_gate(self, fake_env):
        """PREFLIGHT_SKIP_COMPLETENESS=1 must let an incomplete tree through
        the gate (emergency override) -- dry-run then reaches exit 0."""
        tmp_path, env_file, theme_dir = fake_env
        (theme_dir / "readme.txt").unlink()  # deliberately incomplete
        result = run_script(
            "--dry-run",
            env_overrides={
                "ENV_FILE": str(env_file),
                "THEME_DIR_OVERRIDE": str(theme_dir),
                "PREFLIGHT_SKIP_COMPLETENESS": "1",
            },
        )
        assert result.returncode == 0, f"skip override should pass; stderr: {result.stderr}"
        assert "SKIPPED" in result.stdout

    def test_git_error_fails_closed(self, fake_env):
        """A corrupt git index must fail the gate CLOSED with a clear message,
        never silently pass as '0/0 on disk' (bug-260 fail-open). Engages the
        tracked-file check by making the theme dir a real work tree, then
        corrupts .git/index so `git ls-files` errors while
        `rev-parse --is-inside-work-tree` still succeeds."""
        tmp_path, env_file, theme_dir = fake_env
        subprocess.run(["git", "init", "-q", str(theme_dir)], check=True)
        # `git add` creates the index; corrupting it (vs deleting) forces
        # ls-files to error rather than return an empty list.
        subprocess.run(["git", "-C", str(theme_dir), "add", "-A"], check=True)
        (theme_dir / ".git" / "index").write_bytes(b"garbage-not-a-git-index")
        result = run_script(
            "--dry-run",
            env_overrides={"ENV_FILE": str(env_file), "THEME_DIR_OVERRIDE": str(theme_dir)},
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert (
            "cannot verify tracked-file completeness" in combined
        ), f"expected clean fail-closed message, got: {combined}"
        assert (
            "0/0 on disk" not in combined
        ), "gate reported false completeness on a broken index (fail-open regression)"

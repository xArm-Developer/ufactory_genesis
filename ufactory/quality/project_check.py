"""Local quality checks (no remote CI required)."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
import tomllib

import yaml

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.safety.gate import sha256_file


MODES = ("fast", "sim", "sdk-sim", "hardware", "release", "deep")
ROBOT_KEYS = ("xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850", "lite6")

# Evidence from an earlier commit may be reused when HEAD only touches these paths.
EVIDENCE_CARRY_PATH_PREFIXES = (
    "README.md",
    "README_zh.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "docs/",
    "examples/README",
    "examples/pick_place/README",
    "examples/packaging/README",
)

FAST_CHECK_NAMES = (
    "config-assets",
    "ruff-check",
    "ruff-format",
    "mypy-domain",
    "compileall",
    "pytest-fast",
    "pytest-safety-coverage",
    "safety-coverage-threshold",
)

# Validated pins may carry advisories without a compatible fix in this stack.
PIP_AUDIT_IGNORED_VULNS = (
    "PYSEC-2026-139",  # torch 2.10.0
    "CVE-2025-3000",  # torch 2.10.0
    "PYSEC-2025-194",  # torch 2.10.0; alias of CVE-2025-3000 (torch.jit.script, not used)
    "PYSEC-2026-165",  # pillow 11.3.0 (Genesis stack; fix is 12.2.0)
    "PYSEC-2026-2249",  # pillow 11.3.0 (Genesis stack; fix is 12.1.1)
    "PYSEC-2026-2250",  # pillow 11.3.0 (Genesis stack; fix is 12.2.0)
    "PYSEC-2026-2251",  # pillow 11.3.0 (Genesis stack; fix is 12.2.0)
    "PYSEC-2026-2252",  # pillow 11.3.0 (Genesis stack; fix is 12.2.0)
    "PYSEC-2026-2253",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-2254",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-2255",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-2256",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-2257",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-2874",  # pillow 11.3.0 (Genesis stack; fix is 12.2.0)
    "PYSEC-2026-3451",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "PYSEC-2026-3453",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "CVE-2026-54058",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "CVE-2026-59197",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "CVE-2026-59198",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "CVE-2026-59200",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    "CVE-2026-59204",  # pillow 11.3.0 (Genesis stack; fix is 12.3.0)
    # pillow 11.3.0 (Genesis -> moviepy 2.2.1 requires pillow<12.0; fixes require 13.0.0)
    "CVE-2026-96636",
    "CVE-2026-96637",
    "CVE-2026-96638",
    "CVE-2026-96639",
    "CVE-2026-96640",
    "CVE-2026-96641",
    "CVE-2026-96642",
    "CVE-2026-96643",
    "CVE-2026-96644",
    "CVE-2026-96778",
    "CVE-2026-112390",
    "CVE-2026-112391",
    "CVE-2026-112392",
    "CVE-2026-112393",
    "CVE-2026-112395",
    "CVE-2026-112396",
    "CVE-2026-112397",
    "CVE-2026-112398",
    "CVE-2026-112399",
    "CVE-2026-112400",
    "CVE-2026-112401",
    "CVE-2026-112402",
    "CVE-2026-112404",
    "CVE-2026-112406",
    "CVE-2026-112891",
    "CVE-2026-112892",
    "CVE-2026-112893",
    "CVE-2026-112894",
    "CVE-2026-112895",
    "CVE-2026-112896",
    "CVE-2026-112897",
    "CVE-2026-112899",
    "CVE-2026-112900",
    "CVE-2026-112901",
    "CVE-2026-114501",
    "GHSA-78g4-xqwv-q3g8",
    "PYSEC-2026-4327",
    "PYSEC-2026-4332",
)

UV_EXPORT_EXTRAS = ("sim", "real", "rl", "dynamics", "dev")


@dataclass
class CheckResult:
    name: str
    status: str
    duration_s: float
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    reason: str = ""
    output_tail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args: Sequence[str], root: Path) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def package_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml project.version is required")
    return version.strip()


def _path_allowed_for_evidence_carry(path: str) -> bool:
    normalized = path.replace("\\", "/")
    for prefix in EVIDENCE_CARRY_PATH_PREFIXES:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            return True
    return False


def commits_differ_only_by_docs(root: Path, base: str, head: str) -> bool:
    if base == head:
        return True
    try:
        diff = _git(("diff", "--name-only", f"{base}..{head}"), root)
    except (OSError, subprocess.CalledProcessError):
        return False
    paths = [line.strip() for line in diff.splitlines() if line.strip()]
    if not paths:
        return True
    return all(_path_allowed_for_evidence_carry(path) for path in paths)


def _fast_report_complete(report: dict[str, Any]) -> bool:
    if report.get("mode") != "fast" or report.get("passed") is not True:
        return False
    names = {item.get("name") for item in report.get("checks", []) if isinstance(item, dict)}
    return set(FAST_CHECK_NAMES).issubset(names)


def filter_uv_export_lines(text: str) -> list[str]:
    """Drop comments and editable lines from ``uv export`` output."""
    return [
        line for line in text.splitlines() if line.strip() and not line.startswith("#") and not line.startswith("-e ")
    ]


def build_pip_audit_command(requirements_path: Path, *, python: str | None = None) -> list[str]:
    """Audit a fully pinned requirements file without creating a temp venv or downloading wheels."""
    command: list[str] = [
        python or sys.executable,
        "-m",
        "pip_audit",
        "--strict",
        "--no-deps",
        "--disable-pip",
        "--progress-spinner",
        "off",
        "--timeout",
        "30",
        "-r",
        str(requirements_path),
    ]
    for vuln_id in PIP_AUDIT_IGNORED_VULNS:
        command.extend(["--ignore-vuln", vuln_id])
    return command


def compare_lock_to_installed(requirement_lines: Sequence[str]) -> dict[str, Any]:
    """Compare active exact pins to the current environment.

    Installed version drift is a hard failure signal for callers. Packages listed
    in the lock export but not installed are reported only (extras may be absent).
    """
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    mismatches: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    matched = 0
    skipped = 0
    for line in requirement_lines:
        try:
            req = Requirement(line)
        except ValueError:
            skipped += 1
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        pinned: str | None = None
        for spec in req.specifier:
            if spec.operator != "==":
                pinned = None
                break
            pinned = spec.version
        if pinned is None:
            skipped += 1
            continue
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append({"name": req.name, "locked": pinned})
            continue
        try:
            same = Version(installed) == Version(pinned)
        except InvalidVersion:
            same = installed == pinned
        if same:
            matched += 1
        else:
            mismatches.append({"name": req.name, "locked": pinned, "installed": installed})
    return {
        "matched": matched,
        "missing": missing,
        "mismatches": mismatches,
        "skipped": skipped,
    }


class ProjectCheck:
    def __init__(self, root: Path, mode: str) -> None:
        self.root = root
        self.mode = mode
        self.results: list[CheckResult] = []
        try:
            self.commit = _git(("rev-parse", "HEAD"), root)
            self.dirty = bool(_git(("status", "--porcelain"), root))
        except (OSError, subprocess.CalledProcessError):
            self.commit = "unknown"
            self.dirty = True

    def command(
        self,
        name: str,
        command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        record: bool = True,
        cwd: Path | None = None,
    ) -> CheckResult:
        started = time.perf_counter()
        try:
            run = subprocess.run(
                list(command),
                cwd=cwd or self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            output = (run.stdout + "\n" + run.stderr).strip()
            result = CheckResult(
                name=name,
                status="PASS" if run.returncode == 0 else "FAIL",
                duration_s=time.perf_counter() - started,
                command=list(command),
                exit_code=run.returncode,
                reason="" if run.returncode == 0 else f"command exited with {run.returncode}",
                output_tail=output[-12000:],
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = CheckResult(
                name=name,
                status="FAIL",
                duration_s=time.perf_counter() - started,
                command=list(command),
                reason=str(exc),
            )
        if record:
            self.results.append(result)
        return result

    def record(self, name: str, function: Any) -> CheckResult:
        started = time.perf_counter()
        try:
            data = function() or {}
            result = CheckResult(name, "PASS", time.perf_counter() - started, data=data)
        except Exception as exc:
            result = CheckResult(name, "FAIL", time.perf_counter() - started, reason=str(exc))
        self.results.append(result)
        return result

    def incomplete(self, name: str, reason: str) -> None:
        self.results.append(CheckResult(name, "INCOMPLETE", 0.0, reason=reason))

    def config_and_assets(self) -> dict[str, Any]:
        store = RepositoryAssetStore.discover(self.root)
        manifest = store.validate_manifest()
        hashes = manifest.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError("asset manifest sha256 must be a mapping")
        for relative, expected in hashes.items():
            actual = sha256_file(store.require(relative))
            if actual != expected:
                raise ValueError(f"asset hash mismatch: {relative}")
        configs = {key: load_runtime_config(key, asset_store=store).sha256 for key in ROBOT_KEYS}
        if len(set(configs.values())) != len(configs):
            raise ValueError("robot default configurations unexpectedly share a hash")
        return {"manifest_version": manifest["manifest_version"], "config_sha256": configs}

    def fast(self) -> None:
        python = sys.executable
        self.record("config-assets", self.config_and_assets)
        self.command("ruff-check", (python, "-m", "ruff", "check", "."))
        self.command("ruff-format", (python, "-m", "ruff", "format", "--check", "."))
        self.command(
            "mypy-domain",
            (
                python,
                "-m",
                "mypy",
                "--strict",
                "--follow-imports=skip",
                "ufactory/config",
                "ufactory/safety/models.py",
                "ufactory/safety/interfaces.py",
                "ufactory/safety/statistics.py",
                "ufactory/safety/sdk_sim.py",
                "ufactory/safety/gate.py",
                "ufactory/safety/approved.py",
                "ufactory/safety/clock.py",
                "ufactory/trajectory/segments.py",
                "ufactory/trajectory/planner.py",
                "ufactory/trajectory/validation.py",
                "ufactory/trajectory/execution.py",
                "ufactory/trajectory/preflight.py",
            ),
        )
        self.command("compileall", (python, "-m", "compileall", "-q", "ufactory", "examples", "scripts"))
        self.command(
            "pytest-fast",
            (
                python,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not slow and not gpu and not integration and not display and not hardware",
                "--cov=ufactory",
                "--cov-branch",
                "--cov-fail-under=75",
                "--cov-report=term-missing",
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "QD_KERNEL_COVERAGE": "0"},
        )
        self.command(
            "pytest-safety-coverage",
            (
                python,
                "-m",
                "pytest",
                "-q",
                "tests/safety",
                # Prefer --cov=ufactory over --cov=ufactory.safety: the submodule
                # path triggers a numpy extension double-load on Torch-less CI.
                "--cov=ufactory",
                "--cov-branch",
                "--cov-fail-under=0",
                "--cov-report=term-missing",
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "QD_KERNEL_COVERAGE": "0"},
        )
        self.command(
            "safety-coverage-threshold",
            (
                python,
                "-m",
                "coverage",
                "report",
                "--include=ufactory/safety/*",
                "--omit=ufactory/safety/adapters/*",
                "--fail-under=90",
                "--show-missing",
            ),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "QD_KERNEL_COVERAGE": "0"},
        )

    def sim(self) -> None:
        self.fast()
        self.command(
            "pytest-sim",
            (sys.executable, "-m", "pytest", "-q", "-m", "gpu and not slow", "--maxfail=1"),
        )

    def deep(self) -> None:
        """Optional maintainer heavy suite (run ``project-check fast`` first).

        Collects only GPU / integration / display / slow markers — not the full
        unmarked unit suite. Five-robot cabinet CLI evidence belongs in
        ``sdk-sim`` (already parallel by robot IP); in-process GPU stays serial.
        """
        self.command(
            "pytest-deep",
            (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "gpu or integration or display or slow",
                "--maxfail=1",
            ),
        )

    def _run_inventory_command(
        self,
        name: str,
        command: list[str],
        *,
        robot_key: str,
        serial_number: str,
    ) -> CheckResult:
        result = self.command(name, command, record=False)
        result.data.update(robot_key=robot_key, serial_number=serial_number)
        return result

    def _inventory(self, path: Path | None, mode: str, *, confirm_real: bool) -> None:
        if path is None or not path.is_file():
            self.incomplete(f"{mode}-inventory", "an operator-maintained --inventory YAML is required")
            return
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason="invalid inventory schema"))
            return
        robots = data.get("robots")
        if not isinstance(robots, list):
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason="robots must be a list"))
            return
        by_key = {str(item.get("robot_key")): item for item in robots if isinstance(item, dict)}
        missing = sorted(set(ROBOT_KEYS) - set(by_key))
        if missing:
            self.results.append(CheckResult(f"{mode}-inventory", "FAIL", 0.0, reason=f"missing robots: {missing}"))
            return
        if mode == "hardware" and not confirm_real:
            self.results.append(CheckResult("hardware-confirmation", "FAIL", 0.0, reason="--confirm-real is required"))
            return

        # Validate rows first, then run one robot's command matrix per worker so
        # distinct IPs proceed in parallel while the same robot stays serial.
        jobs: list[tuple[str, str, list[list[str]]]] = []
        for key in ROBOT_KEYS:
            serial = str(by_key[key].get("serial_number", "")).strip()
            if len(serial) < 8 or serial == "REQUIRED":
                self.incomplete(f"{mode}-{key}-identity", "inventory requires the complete robot serial number")
                continue
            commands = by_key[key].get(f"{mode}_commands")
            if not isinstance(commands, list) or not commands:
                self.incomplete(f"{mode}-{key}", "inventory has no approved command matrix")
                continue
            normalized: list[list[str]] = []
            invalid = False
            for index, command in enumerate(commands):
                if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                    self.results.append(
                        CheckResult(f"{mode}-{key}-{index}", "FAIL", 0.0, reason="command must be argv list")
                    )
                    invalid = True
                    break
                normalized.append(list(command))
            if not invalid:
                jobs.append((key, serial, normalized))

        if not jobs:
            return

        def _robot_job(item: tuple[str, str, list[list[str]]]) -> list[CheckResult]:
            key, serial, commands = item
            robot_results: list[CheckResult] = []
            for index, command in enumerate(commands):
                robot_results.append(
                    self._run_inventory_command(
                        f"{mode}-{key}-{index}",
                        command,
                        robot_key=key,
                        serial_number=serial,
                    )
                )
            return robot_results

        # Cap workers at robot count; hardware/sdk-sim use distinct IPs per key.
        with ThreadPoolExecutor(max_workers=min(len(jobs), len(ROBOT_KEYS))) as pool:
            futures = [pool.submit(_robot_job, job) for job in jobs]
            for future in as_completed(futures):
                self.results.extend(future.result())

        # Stable report order: robot key then command index.
        inventory_prefix = f"{mode}-"
        inventory_results = [item for item in self.results if item.name.startswith(inventory_prefix)]
        other_results = [item for item in self.results if not item.name.startswith(inventory_prefix)]

        def _sort_key(result: CheckResult) -> tuple[str, int]:
            match = re.fullmatch(rf"{re.escape(mode)}-(.+)-(\d+)", result.name)
            if match is None:
                return (result.name, 0)
            return (match.group(1), int(match.group(2)))

        self.results = other_results + sorted(inventory_results, key=_sort_key)

    def _find_reusable_fast_report(self) -> dict[str, Any] | None:
        evidence_dir = self.root / "reports" / "project-check"
        if not evidence_dir.is_dir():
            return None
        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for path in evidence_dir.glob("*_fast.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("git", {}).get("commit") != self.commit:
                continue
            if not _fast_report_complete(report):
                continue
            mtime = path.stat().st_mtime
            candidates.append((mtime, path, report))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, path, report = candidates[0]
        report = dict(report)
        report["_reused_from"] = str(path)
        return report

    def snapshot_fast(self, *, record: bool = True) -> list[CheckResult]:
        results: list[CheckResult] = []
        reused = self._find_reusable_fast_report()
        if reused is not None:
            results.append(
                CheckResult(
                    "snapshot-fast",
                    "PASS",
                    0.0,
                    reason="reused same-commit fast report",
                    data={"snapshot_report": reused, "reused_from": reused.get("_reused_from")},
                )
            )
            if record:
                self.results.extend(results)
            return results

        # Use a detached worktree so snapshot tests that call ``git ls-files`` /
        # ``git show`` still have repository metadata (``git archive`` does not).
        with tempfile.TemporaryDirectory(prefix="ufactory-release-") as temp_dir:
            snapshot = Path(temp_dir) / "source"
            add = subprocess.run(
                ["git", "worktree", "add", "--detach", str(snapshot), "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if add.returncode != 0:
                results.append(
                    CheckResult("snapshot-fast", "FAIL", 0.0, reason=add.stderr.strip() or add.stdout.strip())
                )
                if record:
                    self.results.extend(results)
                return results
            report_path = Path(temp_dir) / "snapshot-report.json"
            env = {**os.environ, "PYTHONPATH": str(snapshot), "CUDA_VISIBLE_DEVICES": ""}
            try:
                result = self.command(
                    "snapshot-fast",
                    (sys.executable, "-m", "ufactory.quality.project_check", "fast", "--report", str(report_path)),
                    env=env,
                    record=False,
                    cwd=snapshot,
                )
            finally:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(snapshot)],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                )
            if result.status == "PASS" and report_path.is_file():
                result.data["snapshot_report"] = json.loads(report_path.read_text(encoding="utf-8"))
            results.append(result)
        if record:
            self.results.extend(results)
        return results

    def _collect_evidence(self, required: str) -> list[str]:
        evidence_dir = self.root / "reports" / "project-check"
        matches: list[str] = []
        if not evidence_dir.is_dir():
            return matches
        for path in evidence_dir.glob(f"*_{required}.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("passed") is not True:
                continue
            evidence_commit = report.get("git", {}).get("commit")
            if not isinstance(evidence_commit, str) or not evidence_commit:
                continue
            if evidence_commit == self.commit or commits_differ_only_by_docs(self.root, evidence_commit, self.commit):
                matches.append(str(path))
        return matches

    def _export_and_audit(self) -> list[CheckResult]:
        """Frozen uv.lock audit without temp-venv wheel downloads, plus installed drift check."""
        results: list[CheckResult] = []
        with tempfile.TemporaryDirectory(prefix="ufactory-pip-audit-") as temp_dir:
            export_path = Path(temp_dir) / "uv-export.txt"
            filtered_path = Path(temp_dir) / "requirements.txt"
            export_cmd: list[str] = [
                "uv",
                "export",
                "--frozen",
                "--no-annotate",
                "--no-hashes",
            ]
            for extra in UV_EXPORT_EXTRAS:
                export_cmd.extend(["--extra", extra])
            export_cmd.extend(["-o", str(export_path)])
            export = subprocess.run(
                export_cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            if export.returncode != 0:
                results.append(
                    CheckResult(
                        "pip-audit",
                        "FAIL",
                        0.0,
                        reason=(export.stderr or export.stdout or "uv export failed").strip(),
                    )
                )
                return results

            lines = filter_uv_export_lines(export_path.read_text(encoding="utf-8"))
            filtered_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            results.append(self.command("pip-audit", build_pip_audit_command(filtered_path), record=False))

            comparison = compare_lock_to_installed(lines)
            mismatches = comparison["mismatches"]
            if mismatches:
                sample = ", ".join(
                    f"{item['name']} locked={item['locked']} installed={item['installed']}" for item in mismatches[:5]
                )
                more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
                results.append(
                    CheckResult(
                        "lock-installed-match",
                        "FAIL",
                        0.0,
                        reason=f"{len(mismatches)} installed version drift(s): {sample}{more}",
                        data=comparison,
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "lock-installed-match",
                        "PASS",
                        0.0,
                        reason=(
                            f"matched={comparison['matched']} "
                            f"missing={len(comparison['missing'])} "
                            f"(missing extras not failing)"
                        ),
                        data=comparison,
                    )
                )
        return results

    def release(self, version: str | None) -> None:
        expected = package_version(self.root)
        if version != expected:
            self.results.append(
                CheckResult(
                    "release-version",
                    "FAIL",
                    0.0,
                    reason=f"--version must match pyproject.toml ({expected})",
                )
            )
        else:
            self.results.append(CheckResult("release-version", "PASS", 0.0, data={"version": expected}))
        if self.dirty:
            self.results.append(CheckResult("clean-worktree", "FAIL", 0.0, reason="release requires a clean worktree"))
            return
        self.results.append(CheckResult("clean-worktree", "PASS", 0.0))

        # Snapshot (local) and lock audit (network/vuln DB) in parallel; stable append order.
        with ThreadPoolExecutor(max_workers=2) as pool:
            snap_future = pool.submit(self.snapshot_fast, record=False)
            audit_future = pool.submit(self._export_and_audit)
            snapshot_results = snap_future.result()
            audit_results = audit_future.result()
        self.results.extend(snapshot_results)
        self.results.extend(audit_results)

        for required in ("sim", "sdk-sim", "hardware"):
            matches = self._collect_evidence(required)
            if matches:
                self.results.append(CheckResult(f"evidence-{required}", "PASS", 0.0, data={"reports": matches}))
            else:
                self.incomplete(f"evidence-{required}", f"no PASS evidence for commit {self.commit}")

    def report(self) -> dict[str, Any]:
        passed = bool(self.results) and all(result.status == "PASS" for result in self.results)
        return {
            "schema_version": 1,
            "mode": self.mode,
            "passed": passed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": self.commit, "dirty": self.dirty},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "genesis": _version("genesis-world"),
                "torch": _version("torch"),
                "xarm_sdk": _version("xarm-python-sdk"),
                "pinocchio": _version("pin"),
                "coal": _version("coal"),
            },
            "checks": [asdict(result) for result in self.results],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="project-check")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--confirm-real", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    root = RepositoryAssetStore.discover().root
    check = ProjectCheck(root, args.mode)
    if args.mode == "fast":
        check.fast()
    elif args.mode == "sim":
        check.sim()
    elif args.mode == "deep":
        check.deep()
    elif args.mode == "sdk-sim":
        check._inventory(args.inventory, "sdk-sim", confirm_real=False)
    elif args.mode == "hardware":
        check._inventory(args.inventory, "hardware", confirm_real=args.confirm_real)
    else:
        check.release(args.version)
    report = check.report()
    path = args.report
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / "reports" / "project-check" / f"{stamp}_{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"project-check {args.mode}: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report: {path}")
    for result in check.results:
        print(f"  {result.status:10s} {result.name}{': ' + result.reason if result.reason else ''}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

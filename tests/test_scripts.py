"""
tests/test_scripts.py - Verification tests for run.ps1 (PowerShell) and run.sh (Bash) launchers.
"""
import os
import pytest

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..")
RUN_PS1_PATH = os.path.join(SCRIPT_DIR, "run.ps1")
RUN_SH_PATH = os.path.join(SCRIPT_DIR, "run.sh")


def test_launcher_scripts_exist():
    """Verify run.ps1 and run.sh exist on disk."""
    assert os.path.isfile(RUN_PS1_PATH), f"run.ps1 not found at {RUN_PS1_PATH}"
    assert os.path.isfile(RUN_SH_PATH), f"run.sh not found at {RUN_SH_PATH}"


def test_run_ps1_structure_and_features():
    """Verify run.ps1 contains python check, ffmpeg check, venv logic, and argument forwarding."""
    with open(RUN_PS1_PATH, "r", encoding="utf-8") as f:
        ps1_text = f.read()

    # Substantive content
    assert len(ps1_text) > 200

    # Key features
    assert "param(" in ps1_text or "ProcessArgs" in ps1_text
    assert "python" in ps1_text.lower()
    assert "ffmpeg" in ps1_text.lower()
    assert ".venv" in ps1_text
    assert "requirements.txt" in ps1_text
    assert "interview_processor.py" in ps1_text
    assert "inputs" in ps1_text and "outputs" in ps1_text


def test_run_sh_structure_and_features():
    """Verify run.sh contains shebang, python check, ffmpeg check, venv logic, and argument forwarding."""
    with open(RUN_SH_PATH, "r", encoding="utf-8") as f:
        sh_text = f.read()

    # Shebang and basic bash safety
    assert sh_text.startswith("#!/usr/bin/env bash") or sh_text.startswith("#!/bin/bash")
    assert "set -e" in sh_text

    # Key features
    assert "python" in sh_text.lower()
    assert "ffmpeg" in sh_text.lower()
    assert ".venv" in sh_text
    assert "requirements.txt" in sh_text
    assert "interview_processor.py" in sh_text
    assert "inputs" in sh_text and "outputs" in sh_text
    assert "$@" in sh_text or "TARGET_ARGS" in sh_text

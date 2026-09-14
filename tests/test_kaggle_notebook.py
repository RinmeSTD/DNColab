"""
tests/test_kaggle_notebook.py - Unit and structural tests for Interview_AI_Studio_Kaggle.ipynb.
"""
import ast
import json
import os
import pytest

KAGGLE_NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "Interview_AI_Studio_Kaggle.ipynb")


def test_kaggle_notebook_file_exists():
    """Verify Interview_AI_Studio_Kaggle.ipynb exists on disk."""
    assert os.path.isfile(KAGGLE_NOTEBOOK_PATH), f"Kaggle notebook file not found at {KAGGLE_NOTEBOOK_PATH}"


def test_kaggle_notebook_valid_json_structure():
    """Verify notebook is valid JSON with nbformat v4 structure."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert isinstance(nb_data, dict), "Notebook root must be a JSON object"
    assert "cells" in nb_data, "Notebook missing 'cells' key"
    assert "metadata" in nb_data, "Notebook missing 'metadata' key"
    assert "nbformat" in nb_data, "Notebook missing 'nbformat' key"
    assert nb_data["nbformat"] >= 4, f"Expected nbformat >= 4, got {nb_data['nbformat']}"
    assert isinstance(nb_data["cells"], list), "'cells' must be a list"
    assert len(nb_data["cells"]) >= 5, f"Expected at least 5 cells, got {len(nb_data['cells'])}"


def get_kaggle_cell_sources(nb_data):
    """Helper to extract text source for all cells."""
    sources = []
    for cell in nb_data["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        sources.append((cell["cell_type"], src))
    return sources


def test_kaggle_step_1_env_and_rust():
    """Verify Step 1 installs dependencies, checks GPU, and sets up cargo/rust."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    assert len(code_cells) >= 1

    env_cell = code_cells[0]
    assert "torch.cuda.is_available()" in env_cell
    assert "pip install" in env_cell
    assert "deepfilternet" in env_cell


def test_kaggle_step_2_dataset_discovery():
    """Verify Step 2 discovers datasets from /kaggle/input and sets up /kaggle/working."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    discovery_cell = code_cells[1]

    assert "/kaggle/input" in discovery_cell
    assert "/kaggle/working" in discovery_cell


def test_kaggle_step_3_ipywidgets_dashboard():
    """Verify Step 3 creates interactive ipywidgets GUI dashboard."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    widget_cell = code_cells[2]

    assert "ipywidgets" in widget_cell or "widgets." in widget_cell
    assert "FloatSlider" in widget_cell or "slider" in widget_cell.lower()
    assert "Dropdown" in widget_cell
    assert "DeepFilterNet3" in widget_cell
    assert "Checkbox" in widget_cell


def test_kaggle_step_4_batch_queue():
    """Verify Step 4 executes process_batch."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    batch_cell = code_cells[3]

    assert "process_batch" in batch_cell
    assert "summary" in batch_cell.lower()


def test_kaggle_step_5_preview_and_download():
    """Verify Step 5 embeds video and creates downloadable zip in /kaggle/working."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    preview_cell = code_cells[4]

    assert "Video(" in preview_cell
    assert "make_archive" in preview_cell
    assert "zip" in preview_cell.lower()


def test_kaggle_all_code_cells_valid_python_syntax():
    """Verify all code cells compile without SyntaxError."""
    with open(KAGGLE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_kaggle_cell_sources(nb_data)
    for idx, (ctype, code) in enumerate(sources):
        if ctype != "code":
            continue

        sanitized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                sanitized_lines.append(f"# {line}")
            else:
                sanitized_lines.append(line)

        sanitized_code = "\n".join(sanitized_lines)
        try:
            ast.parse(sanitized_code)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in Kaggle code cell {idx}: {e}\nCode:\n{sanitized_code}")

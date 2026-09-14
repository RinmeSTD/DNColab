"""
tests/test_notebook.py - Unit and structural tests for Interview_AI_Studio.ipynb and README.md.
"""
import ast
import json
import os
import re
import pytest

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "Interview_AI_Studio.ipynb")
README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")


def test_notebook_file_exists():
    """Verify Interview_AI_Studio.ipynb exists on disk."""
    assert os.path.isfile(NOTEBOOK_PATH), f"Notebook file not found at {NOTEBOOK_PATH}"


def test_notebook_valid_json_structure():
    """Verify notebook is valid JSON with nbformat v4 structure."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert isinstance(nb_data, dict), "Notebook root must be a JSON object"
    assert "cells" in nb_data, "Notebook missing 'cells' key"
    assert "metadata" in nb_data, "Notebook missing 'metadata' key"
    assert "nbformat" in nb_data, "Notebook missing 'nbformat' key"
    assert nb_data["nbformat"] >= 4, f"Expected nbformat >= 4, got {nb_data['nbformat']}"
    assert isinstance(nb_data["cells"], list), "'cells' must be a list"
    assert len(nb_data["cells"]) >= 5, f"Expected at least 5 cells, got {len(nb_data['cells'])}"

    for idx, cell in enumerate(nb_data["cells"]):
        assert "cell_type" in cell, f"Cell {idx} missing 'cell_type'"
        assert cell["cell_type"] in ("code", "markdown", "raw"), f"Cell {idx} invalid cell_type: {cell['cell_type']}"
        assert "source" in cell, f"Cell {idx} missing 'source'"
        assert "metadata" in cell, f"Cell {idx} missing 'metadata'"
        if cell["cell_type"] == "code":
            assert "outputs" in cell, f"Code cell {idx} missing 'outputs'"
            assert "execution_count" in cell, f"Code cell {idx} missing 'execution_count'"


def get_cell_sources(nb_data):
    """Helper to extract text source for all cells."""
    sources = []
    for cell in nb_data["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        sources.append((cell["cell_type"], src))
    return sources


def test_cell_1_environment_and_dependencies():
    """Verify Cell 1 installs dependencies and checks GPU runtime."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]
    assert len(code_cells) >= 1, "No code cells found"

    # Search for environment setup cell
    env_cell = None
    for code in code_cells:
        if "pip install" in code and ("torch" in code or "deepfilter" in code or "requirements.txt" in code):
            env_cell = code
            break

    assert env_cell is not None, "Could not find environment & dependency setup cell"
    assert "pip install" in env_cell
    assert "torch.cuda.is_available()" in env_cell
    assert "get_device_name" in env_cell or "device_name" in env_cell or "cuda" in env_cell.lower()


def test_cell_2_storage_and_drive_mount():
    """Verify Cell 2 contains Google Drive mount logic and folder configuration."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]

    drive_cell = None
    for code in code_cells:
        if "google.colab" in code and "drive" in code:
            drive_cell = code
            break

    assert drive_cell is not None, "Could not find storage / Google Drive mount code cell"
    assert "drive.mount" in drive_cell
    assert "input" in drive_cell.lower()
    assert "output" in drive_cell.lower()


def test_cell_3_configuration_form_params():
    """Verify Cell 3 implements interactive Colab @param form fields with required types/sliders."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]

    config_cell = None
    for code in code_cells:
        if "@param" in code and "DENOISE_ENGINE" in code:
            config_cell = code
            break

    assert config_cell is not None, "Could not find configuration form cell with @param annotations"

    # Required form parameters from brief
    required_params = [
        "INPUT_DIR",
        "OUTPUT_DIR",
        "DENOISE_ENGINE",
        "MIN_SILENCE_SEC",
        "PADDING_SEC",
        "CROSSFADE_MS",
        "NORMALIZE_AUDIO",
        "EXPORT_TIMELINE",
        "USE_GPU",
        "OVERWRITE",
    ]

    for param in required_params:
        assert param in config_cell, f"Parameter '{param}' missing in configuration form cell"

    # Verify Colab @param form syntax annotations
    assert re.search(r"DENOISE_ENGINE\s*=\s*['\"][^'\"]+['\"]\s*#\s*@param\s*\[.*DeepFilterNet3.*\]", config_cell), \
        "DENOISE_ENGINE must have Colab dropdown @param with DeepFilterNet3"
    assert re.search(r"MIN_SILENCE_SEC\s*=\s*[0-9.]+\s*#\s*@param\s*\{.*slider.*min.*0\.2.*max.*2\.0.*\}", config_cell), \
        "MIN_SILENCE_SEC must have Colab slider @param [0.2, 2.0]"
    assert re.search(r"PADDING_SEC\s*=\s*[0-9.]+\s*#\s*@param\s*\{.*slider.*min.*0\.05.*max.*0\.5.*\}", config_cell), \
        "PADDING_SEC must have Colab slider @param [0.05, 0.5]"
    assert re.search(r"CROSSFADE_MS\s*=\s*[0-9]+\s*#\s*@param\s*\{.*slider.*min.*10.*max.*100.*\}", config_cell), \
        "CROSSFADE_MS must have Colab slider @param [10, 100]"
    assert re.search(r"NORMALIZE_AUDIO\s*=\s*(True|False)\s*#\s*@param\s*\{.*boolean.*\}", config_cell), \
        "NORMALIZE_AUDIO must have Colab boolean @param"
    assert re.search(r"EXPORT_TIMELINE\s*=\s*(True|False)\s*#\s*@param\s*\{.*boolean.*\}", config_cell), \
        "EXPORT_TIMELINE must have Colab boolean @param"
    assert re.search(r"USE_GPU\s*=\s*(True|False)\s*#\s*@param\s*\{.*boolean.*\}", config_cell), \
        "USE_GPU must have Colab boolean @param"
    assert re.search(r"OVERWRITE\s*=\s*(True|False)\s*#\s*@param\s*\{.*boolean.*\}", config_cell), \
        "OVERWRITE must have Colab boolean @param"


def test_cell_4_run_batch_queue():
    """Verify Cell 4 executes process_batch and formats summary metrics."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]

    batch_cell = None
    for code in code_cells:
        if "process_batch" in code:
            batch_cell = code
            break

    assert batch_cell is not None, "Could not find batch execution cell calling process_batch"
    assert "summary" in batch_cell.lower()
    # Check for duration saved / silence cut % / speed factor reporting
    assert "silence" in batch_cell.lower()
    assert "speed" in batch_cell.lower() or "fps" in batch_cell.lower() or "factor" in batch_cell.lower()


def test_cell_5_preview_and_download():
    """Verify Cell 5 provides video preview and zip download functionality."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    code_cells = [src for ctype, src in sources if ctype == "code"]

    preview_cell = None
    for code in code_cells:
        if "Video(" in code or "IPython.display" in code or "files.download" in code or "zipfile" in code or "shutil.make_archive" in code:
            preview_cell = code
            break

    assert preview_cell is not None, "Could not find preview & download cell"
    assert "Video" in preview_cell
    assert "download" in preview_cell.lower() or "zip" in preview_cell.lower()


def test_all_code_cells_valid_python_syntax():
    """Verify that Python code in every code cell compiles without SyntaxError."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    sources = get_cell_sources(nb_data)
    for idx, (ctype, code) in enumerate(sources):
        if ctype != "code":
            continue

        # Filter out shell commands and IPython magics (!, %, %%) for AST syntax parsing
        sanitized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                # Replace with comment or pass so line numbering and block structure are preserved
                sanitized_lines.append(f"# {line}")
            else:
                sanitized_lines.append(line)

        sanitized_code = "\n".join(sanitized_lines)
        try:
            ast.parse(sanitized_code)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in code cell {idx}: {e}\nCode:\n{sanitized_code}")


def test_readme_structure_and_documentation():
    """Verify README.md contains project overview, architecture diagram, quickstart, NLE guide, and parameters."""
    assert os.path.isfile(README_PATH), f"README.md not found at {README_PATH}"

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    # Substantive length
    assert len(readme) > 1000, "README.md is too short"

    # Architecture diagram (mermaid or ascii)
    assert "```mermaid" in readme or "+---" in readme or "|---" in readme, "Missing architecture diagram"

    # Google Colab 1-click badge / link
    assert "colab.research.google.com" in readme, "Missing Google Colab badge or link"

    # Local CLI installation and usage guide
    assert "pip install" in readme, "Missing local pip install instructions"
    assert "interview_processor.py" in readme, "Missing CLI usage instructions"

    # NLE Integration Tutorial for Adobe Premiere Pro and DaVinci Resolve
    assert "Adobe Premiere Pro" in readme or "Premiere Pro" in readme, "Missing Premiere Pro instructions"
    assert "DaVinci Resolve" in readme, "Missing DaVinci Resolve instructions"
    assert ".xml" in readme or "XML" in readme, "Missing XML import instructions"
    assert ".edl" in readme or "EDL" in readme, "Missing EDL import instructions"
    assert "handle" in readme.lower() or "expand" in readme.lower() or "trim" in readme.lower(), \
        "Missing tutorial on expanding/contracting cuts with handles"

    # Configuration parameters reference table
    assert "Parameter" in readme or "Option" in readme or "Argument" in readme, "Missing parameter reference table"
    assert "DeepFilterNet3" in readme
    assert "min_silence" in readme or "MIN_SILENCE" in readme or "--min-silence" in readme

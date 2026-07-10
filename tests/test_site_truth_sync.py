"""Guards that public site claims stay synchronized with source facts."""

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_MODEL_INTERFACE = ROOT / "core" / "brainscore_core" / "model_interface.py"
CORE_CONTRACT = ROOT / "core" / "brainscore_core" / "contract.py"
CORE_BRAINSCORE_MODEL = ROOT / "core" / "brainscore_core" / "brainscore_model.py"
UNIFIED_SCORE = ROOT / "unified" / "brainscore" / "__init__.py"
MEMORY = ROOT / "core" / "brainscore_core" / "memory.py"
ARCH_HTML = ROOT / "unified" / "website" / "architecture.html"
ARCH_JS = ROOT / "unified" / "website" / "architecture.js"
INDEX_HTML = ROOT / "unified" / "website" / "index.html"
DATA_JS = ROOT / "unified" / "website" / "data.js"
README = ROOT / "unified" / "README.md"


def _read(path):
    return path.read_text(encoding="utf-8")


def _source_modality_priority():
    source = _read(CORE_BRAINSCORE_MODEL)
    match = re.search(r"MODALITY_PRIORITY:[^\n=]+=\s*(\([^\)]*\))", source)
    assert match, "source MODALITY_PRIORITY not found"
    return list(ast.literal_eval(match.group(1)))


def _site_modality_priority():
    site = _read(ARCH_JS)
    match = re.search(r"const MODALITY_PRIORITY = (\[[^\]]*\])", site)
    assert match, "site MODALITY_PRIORITY not found"
    return list(ast.literal_eval(match.group(1)))


def _source_input_event_order():
    source = _read(CORE_BRAINSCORE_MODEL)
    match = re.search(r"_INPUT_HANDLERS:[^\n]+=\s*\[(.*?)\]", source, re.S)
    assert match, "source _INPUT_HANDLERS not found"
    return re.findall(r"\((\w+),\s*'_", match.group(1))


def test_architecture_uses_subject_as_primary_abc():
    contract_source = _read(CORE_CONTRACT)
    facade_source = _read(CORE_MODEL_INTERFACE)
    html = _read(ARCH_HTML)
    js = _read(ARCH_JS)

    assert "class Subject(ABC):" in contract_source
    assert "UnifiedModel = Subject" in contract_source
    assert "Subject" in facade_source
    assert "UnifiedModel" in facade_source
    assert "The <code>Subject</code> ABC" in html
    assert "class Subject(ABC):" in html
    assert "Subject.process" in js
    assert "The <code>UnifiedModel</code> ABC" not in html
    assert "class UnifiedModel(ABC):" not in html
    assert "UnifiedModel.process" not in js


def test_architecture_dispatch_facts_match_source():
    html = _read(ARCH_HTML)

    assert _site_modality_priority() == _source_modality_priority()
    event_order = _source_input_event_order()
    assert event_order == ["StateChange", "EnvironmentStep", "Message"]
    assert f"({', '.join(event_order)})" in html
    assert "warn_ambiguous_default(modality)" in html


def test_architecture_preflight_claims_match_source():
    score_source = _read(UNIFIED_SCORE)
    memory_source = _read(MEMORY)
    html = _read(ARCH_HTML)
    js = _read(ARCH_JS)

    assert "check_compatibility(model, benchmark)" in score_source
    assert "check_memory(model, benchmark)" in score_source
    # the pre-flight is now an extraction-PLAN estimate, not a full-set probe
    assert "Extraction-plan memory pre-check" in memory_source
    assert "estimated_metric_memory = estimate_metric_memory" in memory_source
    assert "ExecutionPlan" in memory_source
    # the website must describe the plan estimate, not the old full-set probe
    html_flat = " ".join(html.split())   # collapse wrapping whitespace
    assert "extraction plan" in html_flat
    assert "host RAM" in html_flat
    assert "host-RAM extraction-plan estimate" in js
    assert "runs the benchmark stimulus set" not in html   # the retired claim


def test_capability_status_matrix_is_published():
    data = _read(DATA_JS)
    index = _read(INDEX_HTML)
    readme = _read(README)

    assert '"capability_status"' in data
    assert 'id="capability-status-table"' in index
    assert "## Capability Status Matrix" in readme
    for status in ["validated", "structurally-tested", "EC2-only", "demo-only"]:
        assert status in data
        assert status in readme

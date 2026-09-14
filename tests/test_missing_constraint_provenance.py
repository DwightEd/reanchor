import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).with_name("missing_constraint_provenance.py")
if not path.exists():
    path = Path(__file__).resolve().parents[1] / "src/decoding/missing_constraint_provenance.py"
spec = importlib.util.spec_from_file_location("missing_constraint_provenance", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_value_edit_preserves_stage_and_history():
    p = "Before removing: cook the onions in the beer mixture for 10 to 12 minutes. After removing: cook the onions."
    changed = module.replace_value(p, 16)
    assert changed == p.replace("12 minutes", "16 minutes")
    assert module.replace_value(p, 12) == p


def test_value_edit_refuses_ambiguous_or_absent_claim():
    sentence = "cook the onions in the beer mixture for 10 to 12 minutes."
    for p in ("no timed claim", sentence + sentence):
        with pytest.raises(ValueError):
            module.replace_value(p, 18)


def test_provenance_values_are_fixed_before_model_run():
    with pytest.raises(ValueError):
        module.replace_value("cook the onions in the beer mixture for 10 to 12 minutes.", 14)

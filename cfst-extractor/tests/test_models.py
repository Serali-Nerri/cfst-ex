"""Tests for Pydantic output schema models."""

from cfst_extractor.models import PaperOutput, RefInfo, Specimen


def test_specimen_rounding():
    s = Specimen(specimen_label="SC-1", fc_value=35.123, fy=345.678, n_exp=1650.999)
    assert s.fc_value == 35.12
    assert s.fy == 345.68
    assert s.n_exp == 1651.0


def test_paper_output_invalid():
    output = PaperOutput()
    output.invalid("Not a CFST paper")
    assert output.is_valid is False
    assert output.reason == "Not a CFST paper"
    assert output.Group_A == []
    assert output.Group_B == []
    assert output.Group_C == []


def test_paper_output_valid():
    output = PaperOutput(
        is_valid=True,
        reason="Valid CFST data",
        ref_info=RefInfo(title="Test", authors=["A"], journal="J", year=2024),
        Group_B=[
            Specimen(specimen_label="SC-1", fc_value=35.0, fy=345.0, n_exp=1650.0),
        ],
    )
    d = output.model_dump()
    assert d["is_valid"] is True
    assert len(d["Group_B"]) == 1
    assert d["Group_B"][0]["specimen_label"] == "SC-1"
    assert d["Group_B"][0]["fcy150"] == ""


def test_specimen_defaults():
    s = Specimen(specimen_label="X-1")
    assert s.r_ratio == 0.0
    assert s.fcy150 == ""
    assert s.ref_no == ""
    assert s.e1 == 0.0
    assert s.e2 == 0.0

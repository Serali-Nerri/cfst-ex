"""Tests for unit detection and conversion."""

from cfst_extractor.knowledge.units import (
    bulk_detect_force_unit,
    convert_value,
    detect_unit_in_header,
    is_plausible,
)


def test_detect_unit_in_header():
    assert detect_unit_in_header("Nu (kN)") == "kN"
    assert detect_unit_in_header("fc/MPa") == "MPa"
    assert detect_unit_in_header("t [mm]") == "mm"
    assert detect_unit_in_header("fy (N/mm2)") == "MPa"
    assert detect_unit_in_header("specimen") is None


def test_convert_value():
    # N -> kN
    assert abs(convert_value(1000, "N", "kN") - 1.0) < 0.001
    # MN -> kN
    assert abs(convert_value(0.85, "MN", "kN") - 850.0) < 0.001
    # cm -> mm
    assert abs(convert_value(10, "cm", "mm") - 100.0) < 0.001


def test_bulk_detect_force_unit():
    assert bulk_detect_force_unit([200000, 300000, 150000]) == "N"
    assert bulk_detect_force_unit([0.5, 0.8, 0.3]) == "MN"
    assert bulk_detect_force_unit([200, 300, 150]) is None
    assert bulk_detect_force_unit([]) is None


def test_is_plausible():
    assert is_plausible("fc_value", 35.0) is True
    assert is_plausible("fc_value", 0.5) is False
    assert is_plausible("fc_value", 300.0) is False
    assert is_plausible("n_exp", 1650.0) is True
    assert is_plausible("n_exp", 0.5) is False

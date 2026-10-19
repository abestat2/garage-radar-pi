"""Unit tests: the coverage badge generator used by CI."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("coverage_badge", Path(__file__).resolve().parents[1] / "scripts" / "coverage_badge.py")
badge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(badge)


def write_xml(path, **attrs):
    path.write_text("<coverage " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "></coverage>")
    return path


def test_percent_counts_branches_like_pytest_cov(tmp_path):
    xml = write_xml(tmp_path / "c.xml", **{"line-rate": 1.0, "lines-valid": 90, "lines-covered": 90,
                                           "branches-valid": 10, "branches-covered": 0})
    assert badge.percent(xml) == 90  # not the 100% the line-rate alone suggests


def test_percent_falls_back_to_line_rate_without_counts(tmp_path):
    assert badge.percent(write_xml(tmp_path / "c.xml", **{"line-rate": 0.834})) == 83


@pytest.mark.parametrize("pct,colour", [(100, "#4c1"), (95, "#4c1"), (94, "#97ca00"), (90, "#97ca00"),
                                        (80, "#dfb317"), (75, "#dfb317"), (60, "#fe7d37"), (59, "#e05d44")])
def test_colour_bands(pct, colour):
    assert badge.color(pct) == colour


def test_svg_and_cli_write_the_badge(tmp_path, monkeypatch):
    xml = write_xml(tmp_path / "c.xml", **{"line-rate": 0.5, "lines-valid": 10, "lines-covered": 5,
                                           "branches-valid": 0, "branches-covered": 0})
    out = tmp_path / "nested" / "coverage.svg"
    monkeypatch.setattr("sys.argv", ["coverage_badge.py", str(xml), str(out)])
    runpy = importlib.import_module("runpy")
    runpy.run_path(str(Path(badge.__file__)), run_name="__main__")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<svg") and "coverage: 50%" in text and "#e05d44" in text

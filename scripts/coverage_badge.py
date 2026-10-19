"""Write a flat SVG coverage badge from coverage.xml.  Usage: coverage_badge.py coverage.xml badges/coverage.svg"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def percent(xml_path):
    """Total coverage as pytest-cov reports it (lines + branches when branch coverage is on)."""
    root = ET.parse(xml_path).getroot()
    lines, branches = int(root.get("lines-valid", 0)), int(root.get("branches-valid", 0))
    if lines + branches == 0:
        return round(float(root.get("line-rate", 0)) * 100)
    covered = int(root.get("lines-covered", 0)) + int(root.get("branches-covered", 0))
    return round(covered / (lines + branches) * 100)


def color(pct):
    for floor, c in ((95, "#4c1"), (90, "#97ca00"), (75, "#dfb317"), (60, "#fe7d37")):
        if pct >= floor:
            return c
    return "#e05d44"


def svg(pct):
    text = f"{pct}%"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20" role="img" aria-label="coverage: {text}">'
            '<rect width="63" height="20" fill="#555"/>'
            f'<rect x="63" width="36" height="20" fill="{color(pct)}"/>'
            '<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
            '<text x="31.5" y="14">coverage</text>'
            f'<text x="81" y="14">{text}</text></g></svg>\n')


if __name__ == "__main__":
    xml_path, out = sys.argv[1:3]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(svg(percent(xml_path)), encoding="utf-8")

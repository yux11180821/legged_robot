#!/usr/bin/env python3
"""Set a DOCX document's WordprocessingML fonts to Songti/SimSun."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W_NS, "a": A_NS}


def qn(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def set_rfonts(rpr: etree._Element) -> None:
    rfonts = rpr.find("w:rFonts", NS)
    if rfonts is None:
        rfonts = etree.Element(qn(W_NS, "rFonts"))
        rpr.insert(0, rfonts)

    # Use the Chinese display name for East Asian text and SimSun for Latin/CS.
    rfonts.set(qn(W_NS, "ascii"), "SimSun")
    rfonts.set(qn(W_NS, "hAnsi"), "SimSun")
    rfonts.set(qn(W_NS, "eastAsia"), "宋体")
    rfonts.set(qn(W_NS, "cs"), "SimSun")

    # Remove theme bindings so Word does not silently fall back to theme fonts.
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme", "csTheme"):
        rfonts.attrib.pop(qn(W_NS, attr), None)


def ensure_run_rpr(run: etree._Element) -> etree._Element:
    rpr = run.find("w:rPr", NS)
    if rpr is not None:
        return rpr
    rpr = etree.Element(qn(W_NS, "rPr"))
    run.insert(0, rpr)
    return rpr


def ensure_doc_defaults(styles_root: etree._Element) -> None:
    doc_defaults = styles_root.find("w:docDefaults", NS)
    if doc_defaults is None:
        doc_defaults = etree.Element(qn(W_NS, "docDefaults"))
        styles_root.insert(0, doc_defaults)

    rpr_default = doc_defaults.find("w:rPrDefault", NS)
    if rpr_default is None:
        rpr_default = etree.SubElement(doc_defaults, qn(W_NS, "rPrDefault"))

    rpr = rpr_default.find("w:rPr", NS)
    if rpr is None:
        rpr = etree.SubElement(rpr_default, qn(W_NS, "rPr"))

    set_rfonts(rpr)


def patch_word_xml(xml_bytes: bytes, part_name: str) -> tuple[bytes, int]:
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    root = etree.fromstring(xml_bytes, parser)
    changed = 0

    for run in root.xpath(".//w:r", namespaces=NS):
        set_rfonts(ensure_run_rpr(run))
        changed += 1

    for rpr in root.xpath(".//w:rPr", namespaces=NS):
        set_rfonts(rpr)
        changed += 1

    if part_name == "word/styles.xml":
        ensure_doc_defaults(root)
        changed += 1

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=False), changed


def patch_theme_xml(xml_bytes: bytes) -> tuple[bytes, int]:
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    root = etree.fromstring(xml_bytes, parser)
    changed = 0

    for tag in ("majorFont", "minorFont"):
        node = root.find(f".//a:{tag}", NS)
        if node is None:
            continue
        latin = node.find("a:latin", NS)
        if latin is not None:
            latin.set("typeface", "SimSun")
            changed += 1
        ea = node.find("a:ea", NS)
        if ea is not None:
            ea.set("typeface", "宋体")
            changed += 1
        cs = node.find("a:cs", NS)
        if cs is not None:
            cs.set("typeface", "SimSun")
            changed += 1

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=False), changed


def convert(input_path: Path, output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"parts": 0, "font_updates": 0}

    with tempfile.TemporaryDirectory() as td:
        temp_output = Path(td) / output_path.name
        with zipfile.ZipFile(input_path, "r") as zin, zipfile.ZipFile(temp_output, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                new_data = data
                changed = 0

                if item.filename.startswith("word/theme/") and item.filename.endswith(".xml"):
                    try:
                        new_data, changed = patch_theme_xml(data)
                    except etree.XMLSyntaxError:
                        new_data = data
                elif item.filename.startswith("word/") and item.filename.endswith(".xml"):
                    try:
                        new_data, changed = patch_word_xml(data, item.filename)
                    except etree.XMLSyntaxError:
                        new_data = data

                if changed:
                    counts["parts"] += 1
                    counts["font_updates"] += changed

                zout.writestr(item, new_data)

        shutil.copyfile(temp_output, output_path)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    counts = convert(args.input, args.output)
    print(f"OK: {args.output}")
    print(f"parts={counts['parts']} font_updates={counts['font_updates']}")


if __name__ == "__main__":
    main()

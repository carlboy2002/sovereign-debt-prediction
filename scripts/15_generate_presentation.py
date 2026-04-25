"""
15_generate_presentation.py - Generate the two-minute project PPTX deck.

This script creates a widescreen PowerPoint file directly with OpenXML so it
does not require python-pptx or Microsoft Office.

Output:
    presentation/Beyond_Official_Statistics_2min.pptx
"""
import html
import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from config import DATA_PROCESSED, FIGURES, ROOT, TABLES  # noqa: E402


PPTX_PATH = ROOT / "presentation" / "Beyond_Official_Statistics_2min.pptx"
BUILD_DIR = ROOT / "presentation" / "_pptx_build"

SLIDE_W = 12192000
SLIDE_H = 6858000

COLORS = {
    "navy": "19324D",
    "ink": "172033",
    "muted": "5A6578",
    "teal": "0F766E",
    "gold": "B7791F",
    "wine": "9F2842",
    "paper": "F7F8FB",
    "white": "FFFFFF",
    "line": "D8DEE8",
}


def emu(inches):
    return int(inches * 914400)


def esc(text):
    return escape(str(text))


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def make_dirs():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    for path in [
        BUILD_DIR / "_rels",
        BUILD_DIR / "docProps",
        BUILD_DIR / "ppt" / "_rels",
        BUILD_DIR / "ppt" / "slides" / "_rels",
        BUILD_DIR / "ppt" / "slideMasters" / "_rels",
        BUILD_DIR / "ppt" / "slideLayouts" / "_rels",
        BUILD_DIR / "ppt" / "theme",
        BUILD_DIR / "ppt" / "media",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def text_runs(text, size=22, color=COLORS["ink"], bold=False):
    lines = str(text).split("\n")
    paragraphs = []
    for line in lines:
        paragraphs.append(
            f"""<a:p><a:r><a:rPr lang="en-US" sz="{size * 100}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{esc(line)}</a:t></a:r></a:p>"""
        )
    return "".join(paragraphs)


def textbox(shape_id, x, y, w, h, text, size=22, color=COLORS["ink"], bold=False):
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Text {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{text_runs(text, size, color, bold)}</p:txBody>
    </p:sp>
    """


def rect(shape_id, x, y, w, h, fill, line=None, radius=False):
    geom = "roundRect" if radius else "rect"
    line_xml = (
        f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Shape {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr>
    </p:sp>
    """


def bullet_list(shape_id, x, y, w, h, items, size=20):
    paragraphs = []
    for item in items:
        paragraphs.append(
            f"""<a:p><a:pPr marL="285750" indent="-171450"><a:buChar char="&#8226;"/></a:pPr><a:r><a:rPr lang="en-US" sz="{size * 100}"><a:solidFill><a:srgbClr val="{COLORS['ink']}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{esc(item)}</a:t></a:r></a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Bullets {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
    </p:sp>
    """


def picture(shape_id, rel_id, x, y, w, h, name):
    return f"""
    <p:pic>
      <p:nvPicPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
      <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
    </p:pic>
    """


def card(shape_id, x, y, w, h, label, value, note, accent=COLORS["teal"]):
    return (
        rect(shape_id, x, y, w, h, COLORS["white"], COLORS["line"], radius=True)
        + textbox(shape_id + 1, x + emu(0.18), y + emu(0.16), w - emu(0.36), emu(0.25), label.upper(), 10, COLORS["muted"], True)
        + textbox(shape_id + 2, x + emu(0.18), y + emu(0.42), w - emu(0.36), emu(0.38), value, 28, accent, True)
        + textbox(shape_id + 3, x + emu(0.18), y + emu(0.86), w - emu(0.36), h - emu(0.95), note, 12, COLORS["muted"], False)
    )


def slide_xml(shapes):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{COLORS['paper']}"/></a:solidFill></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {shapes}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels(image_rels):
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for rel_id, target in image_rels:
        rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def write_static_parts(n_slides):
    (BUILD_DIR / "[Content_Types].xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {''.join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n_slides + 1))}
</Types>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "_rels" / ".rels").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        encoding="utf-8",
    )
    slide_ids = "".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, n_slides + 1)
    )
    (BUILD_DIR / "ppt" / "presentation.xml").write_text(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n_slides + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>""",
        encoding="utf-8",
    )
    pres_rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n_slides + 1)
    ]
    pres_rels.append(
        f'<Relationship Id="rId{n_slides + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    )
    (BUILD_DIR / "ppt" / "_rels" / "presentation.xml.rels").write_text(
        f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(pres_rels)}</Relationships>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "ppt" / "slideMasters" / "slideMaster1.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "ppt" / "slideMasters" / "_rels" / "slideMaster1.xml.rels").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "ppt" / "slideLayouts" / "slideLayout1.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "ppt" / "slideLayouts" / "_rels" / "slideLayout1.xml.rels").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "ppt" / "theme" / "theme1.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme"><a:themeElements><a:clrScheme name="Custom"><a:dk1><a:srgbClr val="172033"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="19324D"/></a:dk2><a:lt2><a:srgbClr val="F7F8FB"/></a:lt2><a:accent1><a:srgbClr val="0F766E"/></a:accent1><a:accent2><a:srgbClr val="B7791F"/></a:accent2><a:accent3><a:srgbClr val="9F2842"/></a:accent3><a:accent4><a:srgbClr val="5A6578"/></a:accent4><a:accent5><a:srgbClr val="D8DEE8"/></a:accent5><a:accent6><a:srgbClr val="19324D"/></a:accent6><a:hlink><a:srgbClr val="0F766E"/></a:hlink><a:folHlink><a:srgbClr val="9F2842"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Custom"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "docProps" / "core.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Beyond Official Statistics</dc:title><dc:creator>DSPC7100 Final Project Team</dc:creator></cp:coreProperties>""",
        encoding="utf-8",
    )
    (BUILD_DIR / "docProps" / "app.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>OpenXML Generator</Application></Properties>""",
        encoding="utf-8",
    )


def package_pptx():
    if PPTX_PATH.exists():
        PPTX_PATH.unlink()
    with zipfile.ZipFile(PPTX_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in BUILD_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD_DIR).as_posix())
    shutil.rmtree(BUILD_DIR)


def load_results():
    xgb = pd.read_csv(TABLES / "xgboost_nested_comparison.csv")
    rolling = pd.read_csv(TABLES / "xgboost_rolling_validation_summary.csv")
    fairness = pd.read_csv(TABLES / "fairness_by_region.csv")
    panel = pd.read_csv(DATA_PROCESSED / "panel_features.csv")
    return xgb, rolling, fairness, panel


def write_slide(index, shapes, image_rels=None):
    image_rels = image_rels or []
    (BUILD_DIR / "ppt" / "slides" / f"slide{index}.xml").write_text(
        slide_xml(shapes), encoding="utf-8"
    )
    (BUILD_DIR / "ppt" / "slides" / "_rels" / f"slide{index}.xml.rels").write_text(
        slide_rels(image_rels), encoding="utf-8"
    )


def copy_media(items):
    rels = []
    for idx, (source, name) in enumerate(items, start=2):
        target = f"image{idx - 1}.png"
        shutil.copyfile(source, BUILD_DIR / "ppt" / "media" / target)
        rels.append((f"rId{idx}", target, name))
    return rels


def main():
    xgb, rolling, fairness, panel = load_results()
    macro_sat = xgb[xgb["specification"] == "Macro + Satellite"].iloc[0]
    all_features = xgb[xgb["specification"] == "All Features"].iloc[0]
    macro_only = xgb[xgb["specification"] == "Macro Only"].iloc[0]
    rolling_best = rolling.sort_values("mean_avg_precision", ascending=False).iloc[0]
    fairness_gap = fairness["auc"].max() - fairness["auc"].min()

    make_dirs()
    n_slides = 5
    write_static_parts(n_slides)

    # Slide 1
    shapes = rect(2, 0, 0, SLIDE_W, SLIDE_H, COLORS["navy"])
    shapes += textbox(3, emu(0.7), emu(0.75), emu(7.9), emu(1.25), "Beyond Official Statistics", 42, COLORS["white"], True)
    shapes += textbox(4, emu(0.75), emu(2.0), emu(8.2), emu(0.6), "Predicting sovereign debt crises with macro data, satellite nightlights, and news text", 21, "DCE8F5", False)
    shapes += card(10, emu(0.75), emu(3.35), emu(2.65), emu(1.35), "Countries", str(panel["iso3c"].nunique()), "Low- and middle-income economies", COLORS["teal"])
    shapes += card(20, emu(3.65), emu(3.35), emu(2.65), emu(1.35), "Main AP", fmt(macro_sat["avg_precision"]), "Macro + Satellite validation", COLORS["gold"])
    shapes += card(30, emu(6.55), emu(3.35), emu(2.65), emu(1.35), "Best AUC", fmt(all_features["auc"]), "All Features robustness model", COLORS["wine"])
    shapes += textbox(50, emu(0.75), emu(6.05), emu(8.5), emu(0.35), "Columbia SIPA - DSPC7100 Applying Machine Learning - Spring 2026", 13, "DCE8F5", False)
    write_slide(1, shapes)

    # Slide 2
    shapes = textbox(2, emu(0.55), emu(0.35), emu(11.8), emu(0.55), "Problem and Data", 30, COLORS["navy"], True)
    shapes += textbox(3, emu(0.65), emu(1.05), emu(5.45), emu(0.85), "Official crisis-warning systems depend on statistics that arrive late, are revised, and can be strategically reported by distressed governments.", 18, COLORS["ink"], False)
    shapes += rect(5, emu(0.65), emu(2.15), emu(3.65), emu(2.65), COLORS["white"], COLORS["line"], True)
    shapes += textbox(6, emu(0.9), emu(2.38), emu(3.15), emu(0.35), "World Bank WDI", 18, COLORS["navy"], True)
    shapes += bullet_list(7, emu(0.9), emu(2.88), emu(3.1), emu(1.55), ["GDP growth", "Inflation, reserves, debt", "Official baseline"], 15)
    shapes += rect(10, emu(4.65), emu(2.15), emu(3.65), emu(2.65), COLORS["white"], COLORS["line"], True)
    shapes += textbox(11, emu(4.9), emu(2.38), emu(3.15), emu(0.35), "VIIRS Nightlights", 18, COLORS["navy"], True)
    shapes += bullet_list(12, emu(4.9), emu(2.88), emu(3.1), emu(1.55), ["Observed activity proxy", "Growth and volatility", "GDP-light divergence"], 15)
    shapes += rect(15, emu(8.65), emu(2.15), emu(3.65), emu(2.65), COLORS["white"], COLORS["line"], True)
    shapes += textbox(16, emu(8.9), emu(2.38), emu(3.15), emu(0.35), "GDELT News Text", 18, COLORS["navy"], True)
    shapes += bullet_list(17, emu(8.9), emu(2.88), emu(3.1), emu(1.55), ["Media tone", "Conflict and protest events", "Debt and IMF mentions"], 15)
    shapes += textbox(22, emu(0.75), emu(5.45), emu(11.8), emu(0.55), "Unit: country-year panel, 2000-2023. Target: crisis onset within the next three years.", 17, COLORS["ink"], True)
    write_slide(2, shapes)

    # Slide 3
    shapes = textbox(2, emu(0.55), emu(0.35), emu(11.8), emu(0.55), "Approach", 30, COLORS["navy"], True)
    shapes += bullet_list(
        3,
        emu(0.65),
        emu(1.1),
        emu(5.9),
        emu(3.2),
        [
            "Strict lag-only features: no current-year macro, satellite, or text values enter the model.",
            "Nested XGBoost specifications quantify the incremental value of satellite and text signals.",
            "LASSO provides a sparse, interpretable baseline.",
            "SHAP, fairness audit, and misreporting index support interpretation.",
        ],
        18,
    )
    shapes += rect(10, emu(7.0), emu(1.08), emu(5.3), emu(4.2), COLORS["white"], COLORS["line"], True)
    shapes += textbox(11, emu(7.3), emu(1.35), emu(4.7), emu(0.38), "Validation design", 20, COLORS["navy"], True)
    shapes += bullet_list(
        12,
        emu(7.25),
        emu(1.9),
        emu(4.75),
        emu(2.35),
        [
            "Train: years through 2018",
            "Out-of-time validation: 2019-2020",
            "Rolling-origin checks across five historical windows",
            "2021-2023 is right-censored under a three-year target",
        ],
        15,
    )
    shapes += card(25, emu(7.35), emu(4.55), emu(2.25), emu(1.0), "Feature Blocks", "92", "candidate lag-only features", COLORS["teal"])
    shapes += card(35, emu(9.85), emu(4.55), emu(2.05), emu(1.0), "Models", "4", "nested XGBoost specs", COLORS["gold"])
    write_slide(3, shapes)

    # Slide 4 with ROC image
    media = copy_media([(FIGURES / "roc_comparison.png", "ROC")])
    shapes = textbox(2, emu(0.55), emu(0.35), emu(11.8), emu(0.55), "Results", 30, COLORS["navy"], True)
    shapes += card(5, emu(0.65), emu(1.1), emu(2.55), emu(1.2), "Macro Only AP", fmt(macro_only["avg_precision"]), "official baseline", COLORS["muted"])
    shapes += card(10, emu(3.45), emu(1.1), emu(2.55), emu(1.2), "Macro + Satellite AP", fmt(macro_sat["avg_precision"]), "preferred main model", COLORS["teal"])
    shapes += card(15, emu(6.25), emu(1.1), emu(2.55), emu(1.2), "All Features AUC", fmt(all_features["auc"]), "robustness model", COLORS["gold"])
    shapes += card(20, emu(9.05), emu(1.1), emu(2.55), emu(1.2), "Rolling Mean AP", fmt(rolling_best["mean_avg_precision"]), "Macro + Satellite", COLORS["wine"])
    shapes += picture(30, media[0][0], emu(0.85), emu(2.65), emu(5.5), emu(3.35), media[0][2])
    shapes += textbox(40, emu(6.8), emu(2.75), emu(5.2), emu(0.4), "Main interpretation", 20, COLORS["navy"], True)
    shapes += bullet_list(
        41,
        emu(6.85),
        emu(3.25),
        emu(5.15),
        emu(2.15),
        [
            "Satellite signals deliver the most stable improvement in rare-crisis precision.",
            "Text features add signal, but performance is less stable across windows.",
            "All Features ranks well by AUC, but average precision is weaker than Macro + Satellite.",
        ],
        16,
    )
    write_slide(4, shapes, [(media[0][0], media[0][1])])

    # Slide 5
    shapes = textbox(2, emu(0.55), emu(0.35), emu(11.8), emu(0.55), "Next Steps", 30, COLORS["navy"], True)
    shapes += rect(3, emu(0.65), emu(1.15), emu(5.65), emu(4.6), COLORS["white"], COLORS["line"], True)
    shapes += textbox(4, emu(0.95), emu(1.45), emu(5.0), emu(0.38), "What we have now", 20, COLORS["navy"], True)
    shapes += bullet_list(
        5,
        emu(0.9),
        emu(1.95),
        emu(5.0),
        emu(2.55),
        [
            "A reproducible multimodal early-warning pipeline.",
            "A country-level misreporting index requested in TA feedback.",
            "Fairness diagnostics and region-specific threshold checks.",
            "A transparent account of sensor, text, and label limitations.",
        ],
        16,
    )
    shapes += rect(10, emu(6.75), emu(1.15), emu(5.65), emu(4.6), COLORS["white"], COLORS["line"], True)
    shapes += textbox(11, emu(7.05), emu(1.45), emu(5.0), emu(0.38), "Most valuable next steps", 20, COLORS["navy"], True)
    shapes += bullet_list(
        12,
        emu(7.0),
        emu(1.95),
        emu(5.0),
        emu(2.55),
        [
            "Extend crisis labels through 2026 for a clean 2021-2023 test.",
            "Harmonize DMSP and VIIRS to recover a longer satellite history.",
            "Validate misreporting flags with IMF Article IV and case evidence.",
            "Use the model as decision support, not automated surveillance.",
        ],
        16,
    )
    shapes += textbox(20, emu(0.8), emu(6.2), emu(11.4), emu(0.35), "Bottom line: alternative data helps, especially satellite nightlights, but the responsible claim is predictive early warning with transparent limitations.", 16, COLORS["navy"], True)
    write_slide(5, shapes)

    package_pptx()
    print(f"Saved presentation: {PPTX_PATH}")


if __name__ == "__main__":
    main()

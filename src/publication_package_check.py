#!/usr/bin/env python3
"""Validate the FLMSec camera-ready and arXiv v2 publication packages."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "paper_neurips_workshop" / "camera_ready"
ARXIV = ROOT / "paper_arxiv_v2"
RESULTS = ROOT / "results"
STYLE_SHA256 = "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pdf_text(path: Path) -> tuple[PdfReader, str]:
    reader = PdfReader(path)
    return reader, "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    camera_source = (CAMERA / "paper.tex").read_text()
    arxiv_source = (ARXIV / "paper.tex").read_text()
    style_hash = hashlib.sha256((CAMERA / "neurips_2026.sty").read_bytes()).hexdigest()

    require(style_hash == STYLE_SHA256, "Official NeurIPS style file changed")
    require(
        r"\usepackage[dblblindworkshop,final]{neurips_2026}" in camera_source,
        "Camera-ready source is not using workshop final mode",
    )
    require("Anonymous Author" not in camera_source, "Camera-ready source is anonymous")
    require("Rana Muhammad Usman" in camera_source, "Camera-ready author missing")
    require("October 2026" in arxiv_source, "arXiv version date is not frozen")
    require("Rana Muhammad Usman" in arxiv_source, "arXiv author missing")

    camera_pdf, camera_text = pdf_text(CAMERA / "paper.pdf")
    arxiv_pdf, arxiv_text = pdf_text(ARXIV / "paper.pdf")
    require(len(camera_pdf.pages) == 9,
            "Camera-ready PDF should be eight main/reference pages plus appendix")
    require("References" in (camera_pdf.pages[6].extract_text() or ""),
            "Camera-ready references do not begin on page 7")
    require("Operational Details" in (camera_pdf.pages[8].extract_text() or ""),
            "Camera-ready appendix is not on page 9")
    require(len(arxiv_pdf.pages) == 19, "arXiv PDF is not 19 pages")
    require("Anonymous Author" not in camera_text, "Camera-ready PDF is anonymous")
    require("Rana Muhammad Usman" in camera_text, "Camera-ready PDF author missing")
    require("Acknowledgments and Disclosure of Funding" in camera_text,
            "Camera-ready acknowledgment missing")

    arxiv_source = ROOT / "output" / "arxiv_v2" / "arxiv_2606.00914_v2_source.tar.gz"
    with tarfile.open(arxiv_source, "r:gz") as archive:
        ancillary_name = "anc/counterfactual_evidence_audits_artifacts_v1.zip"
        require(ancillary_name in archive.getnames(),
                "arXiv source package is missing the ancillary artifact release")
        ancillary = archive.extractfile(ancillary_name)
        require(ancillary is not None, "could not read ancillary artifact release")
        with zipfile.ZipFile(io.BytesIO(ancillary.read())) as artifact_zip:
            names = artifact_zip.namelist()
            require(any(name.endswith("/ARTIFACTS.md") for name in names),
                    "ancillary release README missing")
            require(not any(name.endswith(".tex") for name in names),
                    "ancillary release must not contain TeX files")

    audit = json.loads((RESULTS / "susceptibility_audit_structured_analysis.json").read_text())
    cross = json.loads((RESULTS / "cross_interface_audit_analysis.json").read_text())
    frontier = json.loads((RESULTS / "codex_frontier_audit_analysis.json").read_text())
    require(round(audit["primary_validation"]["spearman_rho"], 3) == 0.855,
            "Frozen audit result changed")
    require(round(cross["primary_cross_interface"]["spearman_rho"], 3) == 0.750,
            "Frozen cross-interface result changed")
    require(round(frontier["audit_full_prediction"]["spearman_rho"], 3) == 0.951,
            "Frozen frontier result changed")

    for text, label in ((camera_text, "camera-ready"), (arxiv_text, "arXiv")):
        compact = re.sub(r"\s+", "", text)
        require("ρ=.855" in compact or "rho=.855" in compact,
                f"{label} audit claim missing")
        require("ρ=.750" in compact or "rho=.750" in compact,
                f"{label} RAG claim missing")
        require("ρ=.951" in compact or "rho=.951" in compact,
                f"{label} frontier claim missing")
        require("??" not in text, f"{label} has unresolved references")

    print("PUBLICATION_PACKAGES_OK")


if __name__ == "__main__":
    main()

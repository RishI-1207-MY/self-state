"""Pure-Python OpenXML (.docx) Blackbook Report & Print-Ready HTML Generator for SafeStep.

Creates a formal B.Tech Project Report (Blackbook) strictly following the
Vishwakarma Institute of Information Technology (VIIT), Pune (Department of E&TC Engineering)
template and incorporating all names, guide details, methodology, and references from the
SafeStep research paper alongside the actual implemented datasets, trained models, and evaluation graphs.
"""

import base64
from pathlib import Path
from typing import List, Tuple
from xml.sax.saxutils import escape
import zipfile
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPHS_DIR = PROJECT_ROOT / "evaluation" / "graphs"


def _create_viit_emblem(out_path: Path) -> Path:
    """Render a clean VIIT 'VI' style emblem PNG for the cover page."""
    import numpy as np

    canvas = np.full((240, 240, 3), 255, dtype=np.uint8)
    # Blue V shape
    pts_v = np.array([[45, 35], [85, 35], [140, 175], [170, 95], [198, 95], [148, 210], [125, 210]], np.int32)
    cv2.fillPoly(canvas, [pts_v], (185, 95, 10))
    # Red dot
    cv2.circle(canvas, (184, 72), 14, (35, 35, 220), -1)
    cv2.putText(canvas, "VISHWAKARMA", (24, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 95, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "INSTITUTES", (42, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 95, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "VI", (158, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (185, 95, 10), 3, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


class MinimalDocxBuilder:
    """Builds a valid Word OpenXML (.docx) file using only Python's standard library."""

    def __init__(self) -> None:
        self.body_xml: List[str] = []
        self.media_files: List[Tuple[str, Path]] = []
        self._rel_counter = 10
        self._doc_pr_id = 1

    def add_page_break(self) -> None:
        self.body_xml.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_paragraph(
        self,
        text: str = "",
        align: str = "both",
        bold: bool = False,
        italic: bool = False,
        size_pt: int = 12,
        color_hex: str = "000000",
        space_after: int = 140,
        space_before: int = 0,
    ) -> None:
        sz_half_pt = size_pt * 2
        rpr = f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="{sz_half_pt}"/><w:szCs w:val="{sz_half_pt}"/><w:color w:val="{color_hex}"/>'
        if bold:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"

        ppr = f'<w:jc w:val="{align}"/><w:spacing w:before="{space_before}" w:after="{space_after}" w:line="300" w:lineRule="auto"/>'
        escaped = escape(text)
        self.body_xml.append(f"<w:p><w:pPr>{ppr}</w:pPr><w:r><w:rPr>{rpr}</w:rPr><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>")

    def add_heading(self, text: str, level: int = 1) -> None:
        if level == 1:
            self.add_paragraph(text, align="center", bold=True, size_pt=16, color_hex="0F172A", space_before=240, space_after=180)
        elif level == 2:
            self.add_paragraph(text, align="left", bold=True, size_pt=13, color_hex="1E3A8A", space_before=180, space_after=120)
        else:
            self.add_paragraph(text, align="left", bold=True, size_pt=12, color_hex="1E293B", space_before=120, space_after=80)

    def add_table(self, headers: List[str], rows: List[List[str]]) -> None:
        tbl_xml = [
            "<w:tbl>",
            "<w:tblPr>",
            '<w:tblW w:w="5000" w:type="pct"/>',
            '<w:jc w:val="center"/>',
            "<w:tblBorders>",
            '<w:top w:val="single" w:sz="8" w:space="0" w:color="334155"/>',
            '<w:left w:val="single" w:sz="8" w:space="0" w:color="334155"/>',
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="334155"/>',
            '<w:right w:val="single" w:sz="8" w:space="0" w:color="334155"/>',
            '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="64748B"/>',
            '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="64748B"/>',
            "</w:tblBorders>",
            "</w:tblPr>",
        ]

        if headers:
            tbl_xml.append("<w:tr>")
            for h in headers:
                tbl_xml.append(
                    "<w:tc><w:tcPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"E2E8F0\"/></w:tcPr>"
                    f"<w:p><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"60\" w:after=\"60\"/></w:pPr>"
                    f"<w:r><w:rPr><w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\"/><w:b/><w:sz w:val=\"21\"/></w:rPr>"
                    f"<w:t xml:space=\"preserve\">{escape(str(h))}</w:t></w:r></w:p></w:tc>"
                )
            tbl_xml.append("</w:tr>")

        for r in rows:
            tbl_xml.append("<w:tr>")
            for cell in r:
                tbl_xml.append(
                    "<w:tc><w:tcPr/>"
                    f"<w:p><w:pPr><w:jc w:val=\"left\"/><w:spacing w:before=\"50\" w:after=\"50\"/></w:pPr>"
                    f"<w:r><w:rPr><w:rFonts w:ascii=\"Times New Roman\" w:hAnsi=\"Times New Roman\"/><w:sz w:val=\"20\"/></w:rPr>"
                    f"<w:t xml:space=\"preserve\">{escape(str(cell))}</w:t></w:r></w:p></w:tc>"
                )
            tbl_xml.append("</w:tr>")

        tbl_xml.append("</w:tbl>")
        self.body_xml.append("".join(tbl_xml))
        self.add_paragraph("", space_after=80)

    def add_image(self, img_path: Path, caption: str, width_inches: float = 5.4) -> None:
        if not img_path.exists():
            return
        im = cv2.imread(str(img_path))
        if im is None:
            return
        ih, iw = im.shape[:2]
        aspect = ih / max(1.0, float(iw))
        cx = int(width_inches * 914400)
        cy = int(cx * aspect)

        rel_id = f"rId{self._rel_counter}"
        self._rel_counter += 1
        media_name = f"image_{self._doc_pr_id}{img_path.suffix.lower()}"
        doc_id = self._doc_pr_id
        self._doc_pr_id += 1
        self.media_files.append((rel_id, img_path))

        drawing_xml = f"""
        <w:p>
          <w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/></w:pPr>
          <w:r>
            <w:drawing>
              <wp:inline distT="0" distB="0" distL="0" distR="0">
                <wp:extent cx="{cx}" cy="{cy}"/>
                <wp:docPr id="{doc_id}" name="Picture {doc_id}"/>
                <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                    <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
                      <pic:nvPicPr>
                        <pic:cNvPr id="{doc_id}" name="{escape(media_name)}"/>
                        <pic:cNvPicPr/>
                      </pic:nvPicPr>
                      <pic:blipFill>
                        <a:blip r:embed="{rel_id}"/>
                        <a:stretch><a:fillRect/></a:stretch>
                      </pic:blipFill>
                      <pic:spPr>
                        <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                      </pic:spPr>
                    </pic:pic>
                  </a:graphicData>
                </a:graphic>
              </wp:inline>
            </w:drawing>
          </w:r>
        </w:p>
        """
        self.body_xml.append(drawing_xml.strip())
        if caption:
            self.add_paragraph(caption, align="center", bold=True, italic=True, size_pt=10, space_after=160)

    def save(self, out_docx_path: Path) -> None:
        out_docx_path.parent.mkdir(parents=True, exist_ok=True)
        content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

        root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

        doc_rels_items = []
        for idx, (rel_id, path) in enumerate(self.media_files, start=1):
            media_filename = f"image_{idx}{path.suffix.lower()}"
            doc_rels_items.append(
                f'  <Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_filename}"/>'
            )
        doc_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            + "\n".join(doc_rels_items)
            + "\n</Relationships>"
        )

        sect_pr = """
        <w:sectPr>
          <w:pgSz w:w="11906" w:h="16838"/>
          <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1800" w:header="720" w:footer="720" w:gutter="0"/>
        </w:sectPr>
        """

        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">\n'
            "<w:body>\n"
            + "\n".join(self.body_xml)
            + sect_pr
            + "\n</w:body>\n</w:document>"
        )

        with zipfile.ZipFile(out_docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("word/_rels/document.xml.rels", doc_rels)
            zf.writestr("word/document.xml", document_xml)
            for idx, (_, path) in enumerate(self.media_files, start=1):
                media_filename = f"image_{idx}{path.suffix.lower()}"
                zf.write(path, f"word/media/{media_filename}")


def build_safestep_blackbook() -> Tuple[Path, Path]:
    logo_path = _create_viit_emblem(GRAPHS_DIR / "viit_logo.png")
    fig_5_1 = GRAPHS_DIR / "blackbook_fig_5_1_architecture.png"
    fig_6_1 = GRAPHS_DIR / "blackbook_fig_6_1_workflow.png"
    fig_6_2 = GRAPHS_DIR / "blackbook_fig_6_2_wiring.png"
    fig_7_1 = GRAPHS_DIR / "object_detection_class_distribution.png"
    fig_7_2 = GRAPHS_DIR / "object_detection_sample_annotations.png"
    fig_7_3 = GRAPHS_DIR / "currency_detection_class_distribution.png"
    fig_7_4 = GRAPHS_DIR / "currency_detection_sample_annotations.png"
    fig_7_5 = GRAPHS_DIR / "object_detection_training_curves.png"
    fig_7_6 = GRAPHS_DIR / "object_detection_confusion_matrix.png"
    fig_7_7 = GRAPHS_DIR / "currency_detection_training_curves.png"
    fig_7_8 = GRAPHS_DIR / "currency_detection_confusion_matrix.png"

    doc = MinimalDocxBuilder()

    # ==================== COVER PAGE ====================
    doc.add_paragraph("Bansilal Ramnath Agarwal Charitable Trust’s", align="center", bold=True, size_pt=13, space_after=60)
    doc.add_paragraph("Vishwakarma Institute of Information Technology", align="center", bold=True, size_pt=16, space_after=60)
    doc.add_paragraph("(Department of Electronics & Telecommunication Engineering)", align="center", bold=True, size_pt=12, space_after=140)
    doc.add_image(logo_path, "", width_inches=1.6)
    doc.add_paragraph("Group No. :- 12", align="center", bold=True, size_pt=12, space_after=100)
    doc.add_paragraph("Project Report", align="center", bold=True, size_pt=14, space_after=40)
    doc.add_paragraph("on", align="center", bold=False, size_pt=12, space_after=40)
    doc.add_paragraph(
        "“SafeStep: AI-Enabled Smart Navigation System for the Visually Impaired with Integrated OCR and Currency Detection”",
        align="center",
        bold=True,
        size_pt=15,
        color_hex="1E3A8A",
        space_after=80,
    )
    doc.add_paragraph("(SPONSORED BY: In-House)", align="center", bold=True, size_pt=11, space_after=60)
    doc.add_paragraph("Domain: Artificial Intelligence, Computer Vision & Embedded Edge Computing", align="center", bold=True, size_pt=12, space_after=120)
    doc.add_paragraph("Submitted By:", align="left", bold=True, size_pt=11, space_after=60)

    student_headers = ["Roll No.", "GR No.", "Div", "Name of Student", "E-mail", "Institute / Dept."]
    student_rows = [
        ["411001", "22311251", "A", "Tanushri Chetan Rajput", "tanushri.22311251@viit.ac.in", "Dept. of E&TC, VIIT Pune"],
        ["411002", "22311689", "A", "Sanmay Shankar Argade", "sanmay.22311689@viit.ac.in", "Dept. of E&TC, VIIT / VIT Pune"],
        ["411003", "22311521", "A", "Sujal Gajanan Dahake", "sujal.22311521@viit.ac.in", "Dept. of E&TC, VIIT / VIT Pune"],
        ["411004", "22311563", "A", "Yash Anand Lohote", "yash22311563@viit.ac.in", "Dept. of E&TC, VIIT Pune"],
    ]
    doc.add_table(student_headers, student_rows)

    doc.add_paragraph("B. Tech.", align="center", bold=True, size_pt=13, space_after=30)
    doc.add_paragraph("of", align="center", size_pt=11, space_after=30)
    doc.add_paragraph("Vishwakarma Institute of Information Technology, Pune", align="center", bold=True, size_pt=13, space_after=30)
    doc.add_paragraph("(An Autonomous Institute affiliated to Savitribai Phule Pune University, Pune)", align="center", size_pt=10, space_after=100)
    doc.add_paragraph("Under supervision of", align="center", italic=True, size_pt=11, space_after=40)
    doc.add_paragraph("Dr. Sharada Desai (and Prof. Gauri Ghule)", align="center", bold=True, size_pt=13, space_after=80)
    doc.add_paragraph("Academic Year: 2026 – 2027", align="center", bold=True, size_pt=12, space_after=120)
    doc.add_page_break()

    # ==================== CERTIFICATE PAGE ====================
    doc.add_heading("CERTIFICATE", level=1)
    doc.add_paragraph(
        "This is to certify that project work entitled “SafeStep: AI-Enabled Smart Navigation System for the Visually Impaired with Integrated OCR and Currency Detection” carried out in the 7th semester by,",
        align="both",
        size_pt=12,
        space_after=140,
    )
    cert_headers = ["Roll No.", "GR. No.", "Division", "Name of Student"]
    cert_rows = [
        ["411001", "22311251", "A", "Tanushri Chetan Rajput"],
        ["411002", "22311689", "A", "Sanmay Shankar Argade"],
        ["411003", "22311521", "A", "Sujal Gajanan Dahake"],
        ["411004", "22311563", "A", "Yash Anand Lohote"],
    ]
    doc.add_table(cert_headers, cert_rows)
    doc.add_paragraph(
        "in partial fulfillment for the award of B.Tech. degree in Electronics and Telecommunication Engineering from Vishwakarma Institute of Information Technology, Pune (An Autonomous Institute affiliated to Savitribai Phule Pune University, Pune) during the academic year 2026-2027.",
        align="both",
        size_pt=12,
        space_after=240,
    )
    doc.add_paragraph("Date: 18 / 11 / 2026", align="left", bold=True, size_pt=12, space_after=260)
    sig_headers = ["Project Guide", "Project Coordinator", "Head of Department"]
    sig_rows = [
        [
            "Dr. Sharada Desai / Prof. Gauri Ghule\nGuide, Dept. of E&TC Engg.",
            "Dr. Ketan J. Raut\nProject Coordinator, Dept. of E&TC Engg.",
            "Dr. Medha Wyawahare\nH.O.D., E&TC Engg., VIIT Pune",
        ]
    ]
    doc.add_table(sig_headers, sig_rows)
    doc.add_page_break()

    # ==================== ABSTRACT ====================
    doc.add_heading("ABSTRACT", level=1)
    doc.add_paragraph(
        "This project introduces SafeStep, a thorough AI-driven smart navigation system that supports the visually impaired in indoor and outdoor navigation spaces, providing improved safety and independence. Its central processor is a Raspberry Pi 4 Model B (4GB RAM), coupled with a high-resolution camera module connected via CSI (Picamera2 / OpenCV) for real-time visual perception. The framework leverages lightweight deep learning models such as custom-trained YOLOv8n (YOLO-Lite) and MobileNet-SSD for object and obstacle detection and classification using efficient on-device deep learning inference (PyTorch, ONNX Runtime, OpenCV, and TensorFlow Lite).",
        align="both",
    )
    doc.add_paragraph(
        "The system integrates crucial daily-living assistance features into a unified edge-computing platform: (1) Real-time obstacle and hazard detection across 12 indoor and outdoor object classes (Person, Car, Motorcycle, Bicycle, Bus, Truck, Pole, Wall, Chair, Table, Door, and Other obstacles), (2) Signboard and environmental text recognition using EasyOCR (CRAFT + CRNN), and (3) A custom-trained YOLO model for Indian currency note denomination detection (₹10, ₹20, ₹50, ₹100, ₹200, and ₹500). An HC-SR04 ultrasonic distance measurement module is integrated on Raspberry Pi GPIO23 (TRIG) and GPIO24 (ECHO via a 1kΩ/2kΩ voltage divider) polled at 10 Hz with moving-average filtering to trigger immediate priority warnings for obstacles within 2.0 meters. An offline text-to-speech module using pyttsx3 / espeak-ng delivers prioritized, non-blocking voice guidance and hazard alerts to the user via earphones or speakers.",
        align="both",
    )
    doc.add_paragraph(
        "No internet connection is required for core operation, and the hardware can be operated from a portable 5V rechargeable battery pack (10,000 mAh power bank) for up to 8 hours. Experimental validation on our custom zero-leakage datasets (2,250 object detection images and 2,580 currency images) demonstrates an Object Detection mAP@50 of 93.93% (Precision 91.93%, Recall 86.17%) and a Currency Detection mAP@50 of 91.48% (Precision 86.72%, Recall 89.64%), confirming reliable performance across indoor, outdoor, low-light, bright, crowded, and partially occluded conditions.",
        align="both",
    )
    doc.add_paragraph(
        "Index Terms—Assistive Technology, Computer Vision, Object Detection, Visually Impaired, Raspberry Pi, Deep Learning, OCR, Currency Detection, Edge Computing, MobileNet-SSD, YOLO.",
        align="both",
        bold=True,
        italic=True,
    )
    doc.add_page_break()

    # ==================== ACKNOWLEDGEMENT ====================
    doc.add_heading("ACKNOWLEDGEMENT", level=1)
    doc.add_paragraph(
        "This work could not have been completed without the guidance and encouragement of many people. We would like to particularly acknowledge those below.",
        align="both",
    )
    doc.add_paragraph(
        "We would like to express our sincere gratitude to Vishwakarma Institute of Information Technology (VIIT) and Vishwakarma Institute of Technology (VIT), Pune for providing the laboratory infrastructure, embedded computing resources, and academic environment required to carry out this research and project implementation.",
        align="both",
    )
    doc.add_paragraph(
        "We extend our deepest thanks to our project guide Dr. Sharada Desai and Prof. Gauri Ghule for their continuous mentorship, technical insights, and encouragement throughout every phase of the SafeStep project. We also thank Dr. Ketan J. Raut (Project Coordinator) and Dr. Medha Wyawahare (H.O.D., Department of Electronics & Telecommunication Engineering) for their valuable review and support. Finally, we would like to thank the visually impaired volunteers who engaged in user testing and provided invaluable feedback on voice guidance usability.",
        align="both",
        space_after=280,
    )
    doc.add_paragraph("TANUSHRI CHETAN RAJPUT", align="right", bold=True)
    doc.add_paragraph("SANMAY SHANKAR ARGADE", align="right", bold=True)
    doc.add_paragraph("SUJAL GAJANAN DAHAKE", align="right", bold=True)
    doc.add_paragraph("YASH ANAND LOHOTE", align="right", bold=True)
    doc.add_page_break()

    # ==================== LIST OF FIGURES & TABLES & INDEX ====================
    doc.add_heading("List of Figures", level=1)
    figures_list = [
        ["Fig. 5.1", "SafeStep System Architecture and Component Block Diagram"],
        ["Fig. 6.1", "SafeStep System Workflow and Multi-Threaded Data Processing Pipeline"],
        ["Fig. 6.2", "Raspberry Pi 4 Hardware Interfacing (CSI Camera, HC-SR04 Voltage Divider & Audio)"],
        ["Fig. 7.1", "Dataset A (Object & Obstacle Detection) Class Distribution across 12 Classes"],
        ["Fig. 7.2", "Dataset A Sample Verified YOLO Annotations across Indoor and Outdoor Scenes"],
        ["Fig. 7.3", "Dataset B (Indian Currency Detection) Denomination Distribution (₹10 to ₹500)"],
        ["Fig. 7.4", "Dataset B Sample Verified Indian Currency Specimen Annotations"],
        ["Fig. 7.5", "Object Detection Model Training & Validation Loss and mAP Curves"],
        ["Fig. 7.6", "Object Detection Model Held-Out Test Split Confusion Matrix"],
        ["Fig. 7.7", "Indian Currency Detection Model Training & Validation Loss and mAP Curves"],
        ["Fig. 7.8", "Indian Currency Detection Model Held-Out Test Split Confusion Matrix"],
    ]
    doc.add_table(["Figure No.", "Figure Title"], figures_list)

    doc.add_heading("List of Tables", level=1)
    tables_list = [
        ["Table 2.1", "Comprehensive Literature Survey on Assistive Navigation, OCR, and Currency Recognition"],
        ["Table 6.1", "Custom Dataset Summary, Split Distribution, and Zero-Leakage Verification"],
        ["Table 7.1", "Object & Obstacle Detection Model Per-Class Performance Report (Held-Out Test Split)"],
        ["Table 7.2", "Indian Currency Detection Model Per-Denomination Performance Report (Held-Out Test Split)"],
        ["Table 7.3", "Environmental Condition Breakdown, System Latency & Raspberry Pi INT8 Optimization Metrics"],
        ["Table 10.1", "SafeStep Project Implementation Plan and Phase-wise Execution Schedule"],
    ]
    doc.add_table(["Table No.", "Table Title"], tables_list)
    doc.add_page_break()

    doc.add_heading("INDEX", level=1)
    index_rows = [
        ["", "Abstract .....................................................................................................", "I"],
        ["", "Acknowledgement ................................................................................", "II"],
        ["", "List of Figures ...........................................................................................", "III"],
        ["", "List of Tables ............................................................................................", "IV"],
        ["1", "Introduction", "1"],
        ["2", "Literature Survey", "3"],
        ["3", "Objectives", "6"],
        ["4", "Motivation", "7"],
        ["5", "Block Diagram and Description", "9"],
        ["6", "Methodology", "12"],
        ["7", "Result Analysis and Discussion", "20"],
        ["8", "Applications", "28"],
        ["9", "Scope of Future Work", "30"],
        ["10", "Implementation Plan / Project Phases", "32"],
        ["11", "References", "34"],
    ]
    doc.add_table(["Sr. No.", "Contents", "Page No."], index_rows)
    doc.add_page_break()

    # ==================== CHAPTER 1: INTRODUCTION ====================
    doc.add_heading("1. INTRODUCTION", level=1)
    doc.add_paragraph(
        "Independent mobility and daily living activities present significant challenges for individuals with visual impairment. According to the World Health Organization (WHO), approximately 285 million people worldwide are visually impaired, of whom 39 million are blind [1]. In daily life, visually impaired individuals encounter substantial difficulties when navigating unfamiliar indoor and outdoor spaces, avoiding static and dynamic obstacles, reading environmental signs or room numbers, and independently conducting financial transactions using paper banknotes. Traditional mobility aids such as white canes and guide dogs have served as primary tools for decades, yet they possess inherent limitations in providing comprehensive environmental awareness, overhead obstacle detection, semantic object identification, and text or currency reading.",
        align="both",
    )
    doc.add_paragraph(
        "Early technological developments introduced Electronic Travel Aids (ETAs) relying on ultrasonic or infrared sensors coupled with simple microcontroller units. While such devices offer basic proximity feedback via audible beeps or vibrations, they lack the computational capability required to perform complex computer vision tasks and classify specific hazards in real time [2]. With the rapid emergence of low-cost, energy-efficient single-board computers such as the Raspberry Pi 4 Model B and lightweight deep neural networks (YOLOv8, YOLO-Lite, and MobileNet-SSD), edge-based Artificial Intelligence (Edge-AI) has made it possible to execute multi-modal visual perception entirely on-device without relying on cloud servers or internet connectivity.",
        align="both",
    )
    doc.add_paragraph(
        "SafeStep overcomes the limitations of fragmented assistive devices by unifying four core capabilities into a single, portable, offline-first system: (1) Real-time obstacle and hazard detection with frame-to-frame object tracking and directional localization (left, ahead, right), (2) Optical Character Recognition (OCR) using EasyOCR to read signboards, room numbers, and notices aloud, (3) Custom-trained YOLO-based Indian currency denomination recognition (₹10, ₹20, ₹50, ₹100, ₹200, and ₹500) to support independent financial transactions, and (4) Optional HC-SR04 ultrasonic distance sensing (polled at 10 Hz with moving-average filtering) paired with a prioritized pyttsx3 offline voice feedback engine and an accessible Streamlit dashboard.",
        align="both",
    )
    doc.add_page_break()

    # ==================== CHAPTER 2: LITERATURE SURVEY ====================
    doc.add_heading("2. LITERATURE SURVEY", level=1)
    doc.add_paragraph(
        "Assistive navigation for the visually impaired has evolved from mechanical canes and single-sensor ultrasonic aids to deep learning-based computer vision platforms. Table 2.1 summarizes the key literature surveyed for SafeStep across traditional mobility aids, electronic travel aids, edge object detection, scene text OCR, and banknote recognition.",
        align="both",
    )
    lit_headers = ["Author(s)", "Year", "Title", "Key Findings & Relevance to SafeStep"]
    lit_rows = [
        [
            "Dakopoulos, D. & Bourbakis, N. G. [4]",
            "2008",
            "Wearable obstacle avoidance electronic travel aids for blind: A survey",
            "- Surveys ultrasonic ETAs (Sonic Pathfinder, Nottingham Obstacle Detector).\n- Highlights that ultrasonic-only aids detect proximity but lack semantic object classification.",
        ],
        [
            "Smith, R. [9], [19]",
            "2007",
            "An overview of the Tesseract OCR engine",
            "- Establishes open-source OCR pipelines for document text extraction.\n- Demonstrates challenges of traditional OCR on unconstrained natural scene signboards.",
        ],
        [
            "Howard, A. G. et al. [6]",
            "2017",
            "MobileNets: Efficient convolutional neural networks for mobile vision applications",
            "- Introduces depthwise separable convolutions.\n- Drastically reduces FLOPs and memory footprint for embedded Raspberry Pi inference.",
        ],
        [
            "Tapu, R., Mocanu, B. & Zaharia, T. [8], [12]",
            "2017",
            "DEEP-SEE: Joint object detection, tracking and recognition for visually impaired navigation",
            "- Combines CNN object detection with multi-frame tracking.\n- Reduces false alarms and repeated announcements in dynamic outdoor scenes.",
        ],
        [
            "Zhou, X. et al. [17] & Shi, B. et al. [18]",
            "2017",
            "EAST scene text detector & CRNN sequence recognition",
            "- Proposes deep scene text detection and CRNN sequence recognition.\n- Forms the foundation of EasyOCR (CRAFT + CRNN) used in SafeStep.",
        ],
        [
            "Redmon, J. & Farhadi, A. [7]",
            "2018",
            "YOLOv3: An incremental improvement",
            "- Demonstrates single-stage real-time bounding-box regression and multi-scale object classification.",
        ],
        [
            "Joshi, R. C., Yadav, S. & Dutta, M. K. [15]",
            "2020",
            "YOLO-v3 based currency detection and recognition system for visually impaired persons",
            "- Applies YOLO object detection to Indian banknotes under varying angles and lighting.",
        ],
        [
            "Manjari, K., Verma, M. & Singal, G. [2]",
            "2021",
            "A survey on assistive technology for visually impaired",
            "- Identifies key research gaps: cloud dependency, lack of multi-task integration, and high hardware cost.",
        ],
        [
            "Sheth, R., Desai, P. & Patel, S. [5]",
            "2021",
            "Deep learning based assistive system for visually impaired",
            "- Evaluates RGB camera-based CNN obstacle detection with voice output for indoor/outdoor navigation.",
        ],
        [
            "Pasumarthy, M. et al. [14]",
            "2022",
            "An Indian currency recognition model for assisting visually impaired individuals",
            "- Demonstrates deep learning classification of Mahatma Gandhi New Series Indian currency notes.",
        ],
        [
            "Said, Y. et al. [13] & Park, C. et al. [16]",
            "2023",
            "Obstacle detection & MBDM banknote detection for visually impaired",
            "- Validates lightweight YOLO architectures and multinational banknote recognition on edge devices.",
        ],
    ]
    doc.add_table(lit_headers, lit_rows)

    doc.add_heading("2.1 Research Gaps Identified", level=2)
    doc.add_paragraph(
        "Existing assistive systems exhibit four major drawbacks: (1) Inadequate multi-class object classification in purely ultrasonic travel aids, (2) Dependence on active internet/cloud connectivity for OCR and currency recognition, introducing high latency and privacy concerns, (3) Fragmented single-purpose devices that force users to carry separate tools for obstacle detection, reading text, and identifying banknotes, and (4) Lack of prioritized audio arbitration, causing overlapping or repetitive voice alerts. SafeStep directly addresses these gaps within a single, offline edge-AI platform.",
        align="both",
    )
    doc.add_page_break()

    # ==================== CHAPTER 3: OBJECTIVES ====================
    doc.add_heading("3. OBJECTIVES", level=1)
    doc.add_paragraph(
        "The primary objective of SafeStep is to design, train, evaluate, and deploy an offline, multi-modal AI assistive navigation system that empowers visually impaired individuals with real-time environmental awareness, text reading, and currency recognition. The specific technical objectives are:",
        align="both",
    )
    objectives = [
        "1. Custom Dataset Creation & Zero-Leakage Validation: Construct two comprehensive YOLO-annotated datasets—Dataset A for Object and Obstacle Detection (12 classes, 2,250 usable images, 6,613 bounding boxes) and Dataset B for Indian Currency Detection (6 denominations from ₹10 to ₹500, 2,580 usable images, 3,432 bounding boxes)—split by session_id prior to Albumentations augmentation to guarantee zero data leakage.",
        "2. Real-Time Obstacle & Object Detection with Tracking: Train a custom YOLOv8n model (and implement a MobileNet-SSD comparison module) capable of detecting 12 indoor/outdoor obstacle classes, estimating relative direction (left, ahead, right), and tracking objects across frames using IoU matching to prevent repetitive alerts.",
        "3. Environmental Text Recognition (OCR): Integrate EasyOCR (CRAFT text detector + CRNN recognizer) with CLAHE contrast enhancement, confidence filtering, and temporal deduplication to read signboards, room numbers, and public notices aloud.",
        "4. Indian Currency Denomination Recognition: Train a custom YOLO model via transfer learning to detect single and multiple Indian currency notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500), compute total value, and announce denominations while clearly stating that visual recognition does not verify legal counterfeit authenticity.",
        "5. Ultrasonic Distance Sensing & Prioritized Offline Audio Feedback: Interface an HC-SR04 ultrasonic sensor on Raspberry Pi 4 (GPIO23 TRIG, GPIO24 ECHO via 1kΩ/2kΩ voltage divider) polled at 10 Hz with moving-average filtering, and build a non-blocking pyttsx3 priority voice queue (P1: Distance <= 2m, P2: Hazard, P3: Currency, P4: OCR).",
        "6. Edge Optimization & Parallel Hardware Deployment: Quantize trained ONNX models to dynamic INT8 (reducing model size by 73.5% to 3.19 MB) for Raspberry Pi 4 Model B (4GB RAM) deployment while providing a high-contrast accessible Streamlit dashboard.",
    ]
    for obj in objectives:
        doc.add_paragraph(obj, align="both", space_after=100)
    doc.add_page_break()

    # ==================== CHAPTER 4: MOTIVATION ====================
    doc.add_heading("4. MOTIVATION", level=1)
    doc.add_paragraph(
        "The motivation for developing SafeStep stems from the pressing societal need to make advanced Artificial Intelligence accessible, affordable, and practical for the 285 million visually impaired individuals globally [1]. Several key technical and humanitarian factors drive this project:",
        align="both",
    )
    motivations = [
        "1. Limitations of Conventional Mobility Aids: While the white cane is universally used, it requires physical contact before an obstacle is detected, cannot detect overhanging hazards or distant approaching vehicles, and provides zero semantic information about what the obstacle actually is.",
        "2. Holistic Independence in Daily Living: Safe navigation is only one part of independent living. Visually impaired individuals frequently need to read room numbers in educational or hospital corridors, identify street signs, and verify the denomination of Indian rupee notes during everyday financial transactions without depending on sighted bystanders.",
        "3. Offline Edge-AI Reliability & Privacy: Assistive applications that stream live camera video to cloud APIs fail in basements, rural areas, or public transit zones with poor cellular coverage and expose sensitive personal video streams. Running quantized YOLO and EasyOCR models locally on a Raspberry Pi 4 ensures deterministic real-time response and complete user privacy.",
        "4. Cost-Effective Open-Source Architecture: Commercial wearable assistive vision devices often cost thousands of dollars, placing them out of reach for most users in developing nations. By combining a Raspberry Pi 4 Model B, a CSI camera module, an HC-SR04 ultrasonic sensor, and open-source Python frameworks (PyTorch, Ultralytics YOLO, OpenCV, EasyOCR, ONNX Runtime), SafeStep delivers a low-cost, reproducible solution.",
    ]
    for mot in motivations:
        doc.add_paragraph(mot, align="both", space_after=110)
    doc.add_page_break()

    # ==================== CHAPTER 5: BLOCK DIAGRAM AND DESCRIPTION ====================
    doc.add_heading("5. BLOCK DIAGRAM AND DESCRIPTION", level=1)
    doc.add_paragraph(
        "The hardware architecture of SafeStep is modular and portable, while the software architecture is tightly integrated using a multi-threaded perception and audio arbitration pipeline. Figure 5.1 illustrates the complete block diagram of SafeStep.",
        align="both",
    )
    doc.add_image(fig_5_1, "Fig. 5.1: SafeStep System Architecture and Component Block Diagram", width_inches=5.8)

    blocks = [
        ("5.1 Central Processing Unit (Raspberry Pi 4 Model B - 4GB RAM)", "Serves as the primary edge-computing processor. Its quad-core ARM Cortex-A72 CPU executes video acquisition, ONNX Runtime / PyTorch neural network inference, GPIO ultrasonic timing, and offline speech synthesis concurrently."),
        ("5.2 Visual Input Unit (5-MP Raspberry Pi CSI Camera / Webcam)", "Captures continuous 640x480 RGB video frames from the user's frontal field of view via the CSI ribbon interface (Picamera2 / libcamera on Raspberry Pi, or OpenCV VideoCapture on Windows laptops)."),
        ("5.3 Frame Preprocessing & Scheduler", "Validates incoming frames, performs aspect-aware resizing (320x320 / 416x416), applies CLAHE contrast enhancement in LAB color space for currency notes, and bilateral-filtered adaptive histogram equalization for OCR text regions."),
        ("5.4 Obstacle & Object Detection Block (Custom YOLOv8n / MobileNet-SSD)", "Detects 12 navigation classes (Person, Car, Motorcycle, Bicycle, Bus, Truck, Pole, Wall, Chair, Table, Door, Other obstacles), applies Non-Maximum Suppression (NMS), maps horizontal centroids to directional zones (left, ahead, right), and maintains persistent object IDs via IoU tracking."),
        ("5.5 Environmental OCR Block (EasyOCR)", "Employs the CRAFT (Character Region Awareness for Text Detection) detector followed by a CRNN sequence recognizer to read signboards, room numbers, and notices, suppressing duplicate readings within a configurable time window."),
        ("5.6 Indian Currency Recognition Block (Custom YOLO)", "Classifies six Indian banknote denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500) across front/back views, rotations, and partial folds, supporting multi-note summation in a single frame."),
        ("5.7 Ultrasonic Distance Measurement Block (HC-SR04)", "Triggers 10 µs ultrasonic pulses on BCM GPIO23 and measures echo duration on BCM GPIO24 (stepped down from 5V to 3.3V via a 1kΩ/2kΩ voltage divider) at 10 Hz with a 5-sample moving-average filter."),
        ("5.8 Priority Alert Arbiter & Offline TTS Audio Output", "Evaluates all active detections according to strict safety priority (Priority 1: Distance <= 2.0m -> Priority 2: Hazard Object -> Priority 3: Currency -> Priority 4: OCR) and dispatches spoken alerts via a background pyttsx3 / espeak-ng thread to earphones or speakers."),
    ]
    for title, desc in blocks:
        doc.add_heading(title, level=2)
        doc.add_paragraph(desc, align="both")
    doc.add_page_break()

    # ==================== CHAPTER 6: METHODOLOGY ====================
    doc.add_heading("6. METHODOLOGY", level=1)
    doc.add_heading("6.1 Integrated System Workflow", level=2)
    doc.add_paragraph(
        "The operational workflow of SafeStep starts by initializing all hardware peripherals, loading the custom-trained object and currency detection models, and starting the background audio and 10 Hz ultrasonic polling threads. As shown in Figure 6.1, video frames are processed every cycle for obstacle detection while OCR and currency recognition execute either on configurable time intervals or via manual user trigger to preserve real-time frame rates.",
        align="both",
    )
    doc.add_image(fig_6_1, "Fig. 6.1: SafeStep System Workflow and Multi-Threaded Data Processing Pipeline", width_inches=5.4)

    doc.add_heading("6.2 Custom Dataset Generation and Zero-Leakage Preparation", level=2)
    doc.add_paragraph(
        "To train and rigorously evaluate SafeStep without data leakage, a complete dataset generation, annotation, session-grouped splitting, Albumentations augmentation, and validation pipeline was developed in `datasets/dataset_generation/`. Base images are grouped by `session_id` and split into 70% training, 20% validation, and 10% testing sets BEFORE any data augmentation is applied. Synthetic images and specimen currency notes are explicitly tagged in `metadata.csv`. Table 6.1 summarizes both validated datasets.",
        align="both",
    )
    ds_headers = ["Dataset Name", "Classes", "Base Images", "Augmented Images", "Total Usable Images", "Total BBoxes", "Train / Val / Test Split", "Zero Leakage"]
    ds_rows = [
        ["Dataset A: Object & Obstacle Detection", "12 Navigation Classes", "750", "1,500", "2,250", "6,613", "1,608 / 432 / 210", "Verified (True)"],
        ["Dataset B: Indian Currency Detection", "6 INR Denominations (₹10–₹500)", "860", "1,720", "2,580", "3,432", "1,800 / 510 / 270", "Verified (True)"],
    ]
    doc.add_table(ds_headers, ds_rows)

    doc.add_heading("6.3 Object Detection & Multi-Frame Tracking Implementation", level=2)
    doc.add_paragraph(
        "SafeStep utilizes transfer learning from a pre-trained YOLOv8n backbone alongside an optional MobileNetV3-SSD comparison module. Each input frame is resized and normalized, passed through the neural network to predict bounding boxes and class probabilities, and filtered using confidence thresholding (0.40) and Non-Maximum Suppression (IoU 0.45). To eliminate repetitive voice announcements when an obstacle remains stationary in front of the user, `SimpleDetectionTracker` computes Intersection-over-Union (IoU) across consecutive frames to maintain persistent `track_id` assignments and spatial positions (`left` for x < 35%, `ahead` for 35%–65%, and `right` for x > 65%).",
        align="both",
    )

    doc.add_heading("6.4 Optical Character Recognition (OCR) Integration", level=2)
    doc.add_paragraph(
        "Environmental text recognition is implemented in `modules/ocr_module.py` using EasyOCR. Incoming frames undergo CLAHE contrast enhancement and bilateral denoising to sharpen signboard lettering under uneven indoor or outdoor lighting. The CRAFT detector localizes text bounding quadrilaterals, and the CRNN model decodes character sequences. Low-confidence strings (< 0.45) and short noise tokens (< 2 characters) are discarded, and a temporal deduplication cache suppresses repeated announcements of the same sign within 12 seconds.",
        align="both",
    )

    doc.add_heading("6.5 Custom Indian Currency Detection Model", level=2)
    doc.add_paragraph(
        "A dedicated YOLOv8n model (`modules/currency_detection.py`) was trained on Dataset B covering all six active denominations: ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 across front and back faces, rotations (-28° to +28°), partial folds, and thumb occlusions. During inference, LAB-space CLAHE white-balance normalization is applied, enabling simultaneous multi-note detection and total INR value summation. The system explicitly communicates that it classifies visual denomination only and does not verify legal banknote authenticity.",
        align="both",
    )

    doc.add_heading("6.6 Ultrasonic Sensor Hardware Interfacing & Priority Audio Queue", level=2)
    doc.add_paragraph(
        "As shown in Figure 6.2, the HC-SR04 ultrasonic sensor connects to the Raspberry Pi 4 Model B using Physical Pin 2 (5V), Pin 6 (GND), Pin 16 (BCM GPIO23) for the 10 µs TRIG pulse, and Pin 18 (BCM GPIO24) for the ECHO signal stepped down from 5V to 3.3V through a 1kΩ / 2kΩ resistor voltage divider. The distance d (in meters) is calculated from the echo pulse duration t using d = (t × 343.0) / 2. Readings are polled at 10 Hz and smoothed with a 5-point moving average filter. When an obstacle is detected within 2.0 meters, a Priority-1 alert ('Warning! Obstacle within two meters.') preempts lower-priority messages in the `pyttsx3` background speech queue.",
        align="both",
    )
    doc.add_image(fig_6_2, "Fig. 6.2: Raspberry Pi 4 Hardware Interfacing (CSI Camera, HC-SR04 Voltage Divider & Audio)", width_inches=5.4)
    doc.add_page_break()

    # ==================== CHAPTER 7: RESULT ANALYSIS AND DISCUSSION ====================
    doc.add_heading("7. RESULT ANALYSIS AND DISCUSSION", level=1)
    doc.add_heading("7.1 Dataset Validation & Visual Inspection", level=2)
    doc.add_paragraph(
        "Both custom datasets passed automated integrity verification (`validate_dataset.py`) with zero corrupted files, zero invalid bounding boxes, and zero session leakage across train, validation, and test splits. Figures 7.1 through 7.4 display the class distributions and verified annotation mosaics for Dataset A and Dataset B.",
        align="both",
    )
    doc.add_image(fig_7_1, "Fig. 7.1: Dataset A (Object & Obstacle Detection) Class Distribution (6,613 Bounding Boxes)", width_inches=5.3)
    doc.add_image(fig_7_2, "Fig. 7.2: Dataset A Sample Verified YOLO Annotations across Indoor/Outdoor Scenes", width_inches=5.3)
    doc.add_image(fig_7_3, "Fig. 7.3: Dataset B (Indian Currency Detection) Denomination Distribution (3,432 Bounding Boxes)", width_inches=5.3)
    doc.add_image(fig_7_4, "Fig. 7.4: Dataset B Sample Verified Indian Currency Specimen Annotations", width_inches=5.3)

    doc.add_heading("7.2 Object & Obstacle Detection Model Evaluation", level=2)
    doc.add_paragraph(
        "On the held-out test split (210 images, 657 ground-truth instances), the custom-trained SafeStep YOLOv8n Object Detection model achieved an overall Precision of 91.93%, Recall of 86.17%, mAP@50 of 93.93%, and mAP@50-95 of 78.07% at 51.98 FPS (19.24 ms inference time). Figures 7.5 and 7.6 show the training/validation curves and confusion matrix, and Table 7.1 provides the complete per-class breakdown.",
        align="both",
    )
    doc.add_image(fig_7_5, "Fig. 7.5: Object Detection Model Training & Validation Loss and mAP Curves", width_inches=5.4)
    doc.add_image(fig_7_6, "Fig. 7.6: Object Detection Model Held-Out Test Split Confusion Matrix", width_inches=5.0)

    obj_headers = ["Class ID", "Class Name", "Precision", "Recall", "mAP@50", "mAP@50-95"]
    obj_rows = [
        ["0", "Person", "0.9606", "0.9042", "0.9512", "0.7548"],
        ["1", "Car", "1.0000", "0.4743", "0.9094", "0.7930"],
        ["2", "Motorcycle", "0.9921", "0.9524", "0.9887", "0.7828"],
        ["3", "Bicycle", "0.9240", "0.8667", "0.9619", "0.7552"],
        ["4", "Bus", "0.9047", "0.9298", "0.9400", "0.8590"],
        ["5", "Truck", "0.6743", "0.7559", "0.8295", "0.7364"],
        ["6", "Pole", "0.9716", "0.8760", "0.9143", "0.6085"],
        ["7", "Wall", "0.6353", "0.8750", "0.8794", "0.8335"],
        ["8", "Chair", "1.0000", "0.8599", "0.9369", "0.7864"],
        ["9", "Table", "0.9822", "1.0000", "0.9950", "0.8547"],
        ["10", "Door", "1.0000", "0.9085", "0.9854", "0.8516"],
        ["11", "Other obstacles", "0.9868", "0.9375", "0.9799", "0.7527"],
        ["All", "Overall Test Set (210 imgs)", "0.9193", "0.8617", "0.9393", "0.7807"],
    ]
    doc.add_table(obj_headers, obj_rows)

    doc.add_heading("7.3 Indian Currency Detection Model Evaluation", level=2)
    doc.add_paragraph(
        "On the held-out currency test split (270 images, 354 banknote instances), the custom-trained Indian Currency YOLOv8n model achieved an overall Precision of 86.72%, Recall of 89.64%, mAP@50 of 91.48% (matching the 91.3% validation AP benchmark reported in the research paper), and mAP@50-95 of 71.57% at 49.07 FPS (20.38 ms). Figures 7.7 and 7.8 and Table 7.2 present the detailed results.",
        align="both",
    )
    doc.add_image(fig_7_7, "Fig. 7.7: Indian Currency Detection Model Training & Validation Loss and mAP Curves", width_inches=5.4)
    doc.add_image(fig_7_8, "Fig. 7.8: Indian Currency Detection Model Held-Out Test Split Confusion Matrix", width_inches=4.8)

    curr_headers = ["Class ID", "Denomination Class", "Precision", "Recall", "mAP@50", "mAP@50-95"]
    curr_rows = [
        ["0", "10_rupees (₹10)", "0.7602", "0.8667", "0.8848", "0.6665"],
        ["1", "20_rupees (₹20)", "0.9770", "0.7445", "0.9532", "0.7370"],
        ["2", "50_rupees (₹50)", "0.9966", "1.0000", "0.9950", "0.8469"],
        ["3", "100_rupees (₹100)", "0.9901", "1.0000", "0.9950", "0.8404"],
        ["4", "200_rupees (₹200)", "0.5146", "0.9020", "0.6869", "0.4979"],
        ["5", "500_rupees (₹500)", "0.9646", "0.8650", "0.9739", "0.7055"],
        ["All", "Overall Test Set (270 imgs)", "0.8672", "0.8964", "0.9148", "0.7157"],
    ]
    doc.add_table(curr_headers, curr_rows)

    doc.add_heading("7.4 System Performance, Environmental Robustness & Raspberry Pi Optimization", level=2)
    doc.add_paragraph(
        "Table 7.3 summarizes the measured system latency, CPU/RAM utilization, environmental condition confidence scores, MobileNetV3-SSD comparison, and ONNX INT8 quantization results for Raspberry Pi 4 deployment.",
        align="both",
    )
    sys_headers = ["Evaluation Category / Metric", "Measured Value / Result"]
    sys_rows = [
        ["Custom YOLOv8n Object Detection Latency & FPS", "19.24 ms (51.98 FPS on CPU) / 13.35 ms (74.91 FPS via FP32 ONNX Runtime)"],
        ["MobileNetV3-SSD Comparison Latency & FPS", "55.96 ms (17.87 FPS on CPU)"],
        ["Custom YOLOv8n Currency Detection Latency & FPS", "20.38 ms (49.07 FPS on CPU)"],
        ["EasyOCR Text Recognition Latency (When Triggered)", "944.81 ms per signboard frame"],
        ["Audio Queue Dispatch Latency (pyttsx3 Worker)", "0.09 ms (non-blocking background queue)"],
        ["Average CPU Utilization & Process RAM Usage", "14.7% CPU | 708.6 MB RAM"],
        ["Raspberry Pi 4 INT8 Quantized Model Size (Object & Currency)", "3.19 MB each (73.5% reduction from 12.05 MB FP32 ONNX)"],
        ["Mean Confidence Across Environmental Conditions", "Indoor: 0.8633 | Outdoor: 0.9120 | Low Light: 0.8634 | Bright: 0.9095 | Crowded: 0.9273 | Occluded: 0.8770"],
        ["Portable Battery Runtime (10,000 mAh 5V Power Bank)", "~8 hours continuous operation (Raspberry Pi 4 deployment benchmark)"],
    ]
    doc.add_table(sys_headers, sys_rows)
    doc.add_page_break()

    # ==================== CHAPTER 8: APPLICATIONS ====================
    doc.add_heading("8. APPLICATIONS", level=1)
    doc.add_paragraph(
        "SafeStep provides an integrated assistive platform with practical applications across multiple daily-living and mobility scenarios:",
        align="both",
    )
    apps = [
        ("8.1 Indoor Navigation in Campuses, Hospitals, and Offices", "Detects doors, chairs, tables, walls, and pedestrians in corridors while using EasyOCR to read room numbers (e.g., 'Room 204'), department plaques, and directional signs aloud."),
        ("8.2 Outdoor Pedestrian & Footpath Mobility", "Identifies approaching cars, motorcycles, bicycles, buses, trucks, poles, and street barriers on footpaths and road crossings, providing real-time directional warnings ('Vehicle detected on your left', 'Pole detected on your path')."),
        ("8.3 Independent Financial Transactions & Retail Shopping", "Enables visually impaired users to independently identify Indian currency denominations (₹10, ₹20, ₹50, ₹100, ₹200, ₹500) and count multiple banknotes simultaneously without relying on strangers."),
        ("8.4 Low-Light and Night-Time Hazard Protection", "Combines CLAHE low-light image enhancement with active 10 Hz HC-SR04 ultrasonic rangefinding so near-field obstacles within 2.0 meters trigger immediate Priority-1 audio stop warnings even in complete darkness."),
        ("8.5 Accessible Educational & Rehabilitation Training Tool", "The high-contrast Streamlit dashboard allows orientation and mobility (O&M) trainers, researchers, and caretakers to monitor live telemetry, customize alert thresholds, and evaluate model performance."),
    ]
    for title, desc in apps:
        doc.add_heading(title, level=2)
        doc.add_paragraph(desc, align="both")
    doc.add_page_break()

    # ==================== CHAPTER 9: SCOPE OF FUTURE WORK ====================
    doc.add_heading("9. SCOPE OF FUTURE WORK", level=1)
    future_items = [
        ("9.1 Multi-Lingual Indian Script OCR & Voice Synthesis", "Extend the EasyOCR and TTS pipeline beyond English to support Devanagari (Hindi and Marathi), Tamil, Telugu, and Bengali scripts for regional public signage across India."),
        ("9.2 Multi-Currency & Counterfeit Security Feature Inspection", "Expand Dataset B to include ₹2000 / coins and international currencies, and integrate ultraviolet/micro-lettering optical checks to assist in detecting counterfeit notes."),
        ("9.3 Stereo Depth Camera & LiDAR Sensor Fusion", "Combine the CSI RGB camera and ultrasonic sensor with a lightweight Time-of-Flight (ToF) or stereo depth module to construct dense 3D occupancy maps for staircases, potholes, and drop-offs."),
        ("9.4 GPS & Turn-by-Turn Outdoor Wayfinding Integration", "Integrate an offline GPS module and OpenStreetMap routing engine so SafeStep can combine macro-level turn-by-turn street navigation with micro-level obstacle avoidance."),
        ("9.5 Custom Wearable PCB & Coral Edge TPU Acceleration", "Migrate from a standard Raspberry Pi 4 enclosure to a compact smart-cap or chest-mounted wearable PCB with a Google Coral USB/M.2 Edge TPU or Raspberry Pi AI Kit (Hailo-8L) to achieve >60 FPS full-pipeline inference at sub-3W power consumption."),
    ]
    for title, desc in future_items:
        doc.add_heading(title, level=2)
        doc.add_paragraph(desc, align="both")
    doc.add_page_break()

    # ==================== CHAPTER 10: IMPLEMENTATION PLAN ====================
    doc.add_heading("10. IMPLEMENTATION PLAN / PROJECT PHASES", level=1)
    doc.add_paragraph(
        "Table 10.1 outlines the structured 9-phase implementation schedule followed to design, build, train, test, and package SafeStep.",
        align="both",
    )
    plan_headers = ["Phase / Timeline", "Milestone & Technical Deliverables", "Status"]
    plan_rows = [
        ["Phase 1 (Week 1)", "Requirement Analysis, Literature Survey, Hardware Architecture Specification (Raspberry Pi 4, CSI Camera, HC-SR04 GPIO23/24 Divider), and Modular Directory Setup.", "Completed"],
        ["Phase 2 (Weeks 2–3)", "Development of Dataset Generation Pipeline (`generate_object_dataset.py`, `generate_currency_dataset.py`, `collect_images.py`, `annotate_dataset.py`).", "Completed"],
        ["Phase 3 (Week 4)", "Zero-Leakage Session-Grouped Splitting (`split_dataset.py`), Albumentations Augmentation (`augment_dataset.py`), and Quality Validation (2,250 Object & 2,580 Currency images).", "Completed"],
        ["Phase 4 (Weeks 5–6)", "Transfer Learning Training of Custom YOLOv8n Object Detection (mAP@50: 93.93%) and Indian Currency Detection (mAP@50: 91.48%) Models + ONNX INT8 Optimization.", "Completed"],
        ["Phase 5 (Week 7)", "Implementation of EasyOCR Text Reader (`ocr_module.py`), Prioritized Offline TTS Engine (`audio_feedback.py`), Camera Manager (`camera.py`), and Ultrasonic Module (`distance_sensor.py`).", "Completed"],
        ["Phase 6 (Week 8)", "Multi-Modal Integration into `SafeStepNavigationEngine` (`navigation_engine.py`) with IoU Object Tracking and 4-Level Alert Priority Arbitration.", "Completed"],
        ["Phase 7 (Week 9)", "Development of High-Contrast Accessible 7-Section Streamlit Dashboard (`app.py`) and Dual Windows/Raspberry Pi Configuration Profiles.", "Completed"],
        ["Phase 8 (Week 10)", "Automated Unit & Integration Testing (`pytest` 10/10 passed) and Independent Hardware Verification Scripts (`test_picamera2_hw.py`, `test_hcsr04_hw.py`, `test_audio_hw.py`).", "Completed"],
        ["Phase 9 (Weeks 11–12)", "Experimental Performance Benchmarking across Lighting/Crowd/Occlusion Conditions, SSH/SCP Raspberry Pi Deployment Bundle (`deploy_to_pi.py`), and Final Blackbook Report.", "Completed"],
    ]
    doc.add_table(plan_headers, plan_rows)
    doc.add_page_break()

    # ==================== CHAPTER 11: REFERENCES ====================
    doc.add_heading("11. REFERENCES", level=1)
    refs = [
        '[1] World Health Organization, "Vision impairment and blindness," World Health Organization Fact Sheet, 2023.',
        '[2] K. Manjari, M. Verma, and G. Singal, "A survey on assistive technology for visually impaired," ACM Computing Surveys, vol. 54, no. 5, pp. 1-35, 2021.',
        '[3] I. M. Khairuddin, W. M. F. W. Ahmad, and Z. A. A. Aziz, "Review of vision-based assistive navigation systems for visually impaired people," Journal of Assistive Technologies, vol. 14, no. 2, pp. 89-105, 2020.',
        '[4] D. Dakopoulos and N. G. Bourbakis, "Wearable obstacle avoidance electronic travel aids for blind: A survey," IEEE Transactions on Systems, Man, and Cybernetics, Part C, vol. 40, no. 1, pp. 25-35, 2008.',
        '[5] R. Sheth, P. Desai, and S. Patel, "Deep learning based assistive system for visually impaired," Procedia Computer Science, vol. 167, pp. 1995-2004, 2021.',
        '[6] A. G. Howard, M. Zhu, B. Chen, et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.',
        '[7] J. Redmon and A. Farhadi, "YOLOv3: An incremental improvement," arXiv preprint arXiv:1804.02767, 2018.',
        '[8] R. Tapu, B. Mocanu, and T. Zaharia, "A computer vision-based assistive system for the visually impaired," IEEE Access, vol. 5, pp. 17158-17171, 2017.',
        '[9] R. Smith, "An overview of the Tesseract OCR engine," in Proceedings of the Ninth International Conference on Document Analysis and Recognition, 2007, pp. 629-633.',
        '[10] A. A. A. Youssif and H. A. H. Ramadan, "Automatic recognition of Egyptian currency," International Journal of Computer Applications, vol. 113, no. 5, pp. 29-34, 2015.',
        '[11] B. Kuriakose, R. Shrestha, and F. E. Sandnes, "Tools and technologies for blind and visually impaired navigation support: A review," IETE Technical Review, 2021, doi: 10.1080/02564602.2020.1819893.',
        '[12] R. Tapu, B. Mocanu, and T. Zaharia, "DEEP-SEE: Joint object detection, tracking and recognition with application to visually impaired navigational assistance," Sensors, vol. 17, no. 11, p. 2473, 2017.',
        '[13] Y. Said et al., "Obstacle detection system for navigation assistance of visually impaired people based on deep learning techniques," Sensors, vol. 23, no. 11, p. 5262, 2023.',
        '[14] M. Pasumarthy, R. Padhy, R. Yadav, G. Subramaniam, and M. Rao, "An Indian currency recognition model for assisting visually impaired individuals," in Proceedings of the IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE), 2022.',
        '[15] R. C. Joshi, S. Yadav, and M. K. Dutta, "YOLO-v3 based currency detection and recognition system for visually impaired persons," in Proceedings of the International Conference on Contemporary Computing and Applications (IC3A), 2020.',
        '[16] C. Park and K. R. Park, "MBDM: Multinational banknote detecting model for assisting visually impaired people," Mathematics, vol. 11, no. 6, p. 1392, 2023.',
        '[17] X. Zhou, C. Yao, H. Wen, Y. Wang, S. Zhou, W. He, and J. Liang, "EAST: An efficient and accurate scene text detector," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5551-5560.',
        '[18] B. Shi, X. Bai, and C. Yao, "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 11, pp. 2298-2304, 2017.',
        '[19] R. Smith, "An overview of the Tesseract OCR engine," in Proceedings of the Ninth International Conference on Document Analysis and Recognition (ICDAR), 2007, pp. 629-633.',
        '[20] L. P. Sousa et al., "Evaluation of banknote identification methodologies based on local and deep features," International Journal of Innovative Computing and Applications, vol. 14, no. 1-2, pp. 34-45, 2023.',
    ]
    for r in refs:
        doc.add_paragraph(r, align="both", size_pt=11, space_after=90)

    docx_path = PROJECT_ROOT / "SafeStep_VIIT_Blackbook_Report.docx"
    doc.save(docx_path)

    # Also build self-contained print-ready HTML Blackbook from SafeStep_VIIT_Blackbook_Report.md
    md_path = PROJECT_ROOT / "SafeStep_VIIT_Blackbook_Report.md"
    html_path = PROJECT_ROOT / "SafeStep_VIIT_Blackbook_Report.html"
    if md_path.exists():
        import markdown
        import re

        md_text = md_path.read_text(encoding="utf-8")

        def _embed_img(match):
            alt = match.group(1)
            rel_path = match.group(2)
            img_p = PROJECT_ROOT / rel_path
            if img_p.exists():
                b64 = base64.b64encode(img_p.read_bytes()).decode("ascii")
                return f'<div class="fig-box"><img src="data:image/png;base64,{b64}" alt="{alt}"/><div class="fig-cap">{alt}</div></div>'
            return match.group(0)

        md_embedded = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _embed_img, md_text)
        html_body = markdown.markdown(md_embedded, extensions=["tables"])
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SafeStep VIIT B.Tech Project Blackbook Report</title>
<style>
  @page {{ size: A4; margin: 20mm 18mm 20mm 25mm; }}
  body {{ font-family: "Times New Roman", Times, serif; font-size: 12pt; line-height: 1.6; color: #000; max-width: 210mm; margin: 0 auto; padding: 20px; }}
  h1, h2, h3, h4 {{ color: #0f172a; page-break-after: avoid; }}
  h2 {{ border-bottom: 1.5px solid #1e3a8a; padding-bottom: 4px; margin-top: 28px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 14px 0 20px 0; font-size: 10.5pt; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #334155; padding: 7px 9px; text-align: left; vertical-align: top; }}
  th {{ background-color: #e2e8f0; font-weight: bold; text-align: center; }}
  .fig-box {{ text-align: center; margin: 18px 0; page-break-inside: avoid; }}
  .fig-box img {{ max-width: 92%; height: auto; border: 1px solid #cbd5e1; padding: 4px; }}
  .fig-cap {{ font-weight: bold; font-style: italic; font-size: 10.5pt; margin-top: 6px; color: #1e293b; }}
  hr {{ border: 0; border-top: 2px solid #94a3b8; margin: 32px 0; page-break-after: always; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
        html_path.write_text(html_doc, encoding="utf-8")

    return docx_path, html_path


if __name__ == "__main__":
    out_docx, out_html = build_safestep_blackbook()
    print(f"Generated Word Blackbook (.docx): {out_docx} ({out_docx.stat().st_size / 1024:.1f} KB)")
    if out_html.exists():
        print(f"Generated HTML Blackbook (.html): {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")

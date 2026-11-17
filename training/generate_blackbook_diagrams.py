"""Generate architecture, workflow, and hardware wiring diagrams for the SafeStep VIIT Blackbook."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _draw_box(ax, x, y, w, h, text, bg="#e0f2fe", edge="#0369a1", fontsize=9.5, weight="bold"):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=bg,
        edgecolor=edge,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#0f172a",
    )


def _draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2, color="#1e293b"),
    )


def generate_fig_5_1_block_diagram(out_path: Path) -> None:
    """Generate Fig. 5.1: SafeStep System Architecture and Block Diagram."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(
        5.5,
        6.15,
        "Fig. 5.1: SafeStep System Architecture and Component Block Diagram",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    # Left Column: Inputs & Power
    _draw_box(ax, 0.4, 4.4, 2.5, 1.1, "5-MP Raspberry Pi\nCSI Camera Module\n(Picamera2 / OpenCV)", bg="#dbeafe", edge="#1d4ed8")
    _draw_box(ax, 0.4, 2.7, 2.5, 1.1, "HC-SR04 Ultrasonic\nSensor (10 Hz Polling)\nGPIO23 TRIG / GPIO24 ECHO\n(1kΩ + 2kΩ Divider)", bg="#fef3c7", edge="#b45309")
    _draw_box(ax, 0.4, 1.0, 2.5, 1.0, "5V Rechargeable\nPower Bank\n(10,000 mAh Portable)", bg="#dcfce7", edge="#15803d")

    # Center Column: Raspberry Pi 4 Model B Processing Unit
    outer = mpatches.FancyBboxPatch(
        (3.5, 0.6),
        4.1,
        5.1,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#f8fafc",
        edgecolor="#334155",
        linewidth=2.5,
        linestyle="--",
    )
    ax.add_patch(outer)
    ax.text(5.55, 5.42, "Central Processing Unit: Raspberry Pi 4 Model B (4GB RAM)", ha="center", fontsize=10, fontweight="bold", color="#1e293b")

    _draw_box(ax, 3.8, 4.2, 3.5, 0.9, "1. Frame Preprocessing & CLAHE\n(Resizing, Normalization, Contrast)", bg="#e0e7ff", edge="#4338ca")
    _draw_box(ax, 3.8, 3.1, 3.5, 0.9, "2. Obstacle & Object Detection\n(Custom YOLOv8n / MobileNet-SSD)\n12 Navigation Classes + IoU Tracker", bg="#fae8ff", edge="#a21caf")
    _draw_box(ax, 3.8, 2.0, 3.5, 0.9, "3. Environmental OCR Engine\n(EasyOCR: CRAFT + CRNN)", bg="#ffedd5", edge="#c2410c")
    _draw_box(ax, 3.8, 0.9, 3.5, 0.9, "4. Indian Currency Detection\n(Custom YOLO: ₹10, ₹20, ₹50,\n₹100, ₹200, ₹500)", bg="#dcfce7", edge="#15803d")

    # Right Column: Priority Arbiter & Outputs
    _draw_box(ax, 8.2, 3.4, 2.4, 1.6, "Priority Alert Arbiter\n& Cooldown Filter\nP1: Distance <= 2.0m\nP2: Hazard Object\nP3: INR Currency\nP4: OCR Text", bg="#fee2e2", edge="#b91c1c")
    _draw_box(ax, 8.2, 1.9, 2.4, 1.0, "Offline TTS Audio\n(pyttsx3 / espeak-ng)\nEarphones / Speakers", bg="#dbeafe", edge="#1d4ed8")
    _draw_box(ax, 8.2, 0.6, 2.4, 1.0, "Accessible High-Contrast\nStreamlit Dashboard\n(Live HUD & Telemetry)", bg="#f3e8ff", edge="#6d28d9")

    # Arrows
    _draw_arrow(ax, 2.9, 4.95, 3.8, 4.65)
    _draw_arrow(ax, 2.9, 3.25, 8.2, 4.2)
    _draw_arrow(ax, 2.9, 1.5, 3.5, 1.5)
    _draw_arrow(ax, 5.55, 4.2, 5.55, 4.0)
    _draw_arrow(ax, 7.3, 3.55, 8.2, 4.1)
    _draw_arrow(ax, 7.3, 2.45, 8.2, 3.9)
    _draw_arrow(ax, 7.3, 1.35, 8.2, 3.7)
    _draw_arrow(ax, 9.4, 3.4, 9.4, 2.9)
    _draw_arrow(ax, 7.3, 1.2, 8.2, 1.1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_fig_6_1_workflow_diagram(out_path: Path) -> None:
    """Generate Fig. 6.1: SafeStep System Workflow and Data Processing Pipeline."""
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        5.0,
        6.85,
        "Fig. 6.1: SafeStep System Workflow and Multi-Threaded Data Processing Pipeline",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )

    _draw_box(ax, 3.2, 5.9, 3.6, 0.65, "START: Power ON (5V Power Bank)\nInitialize Raspberry Pi 4 & Peripherals", bg="#dcfce7", edge="#15803d")
    _draw_box(ax, 3.2, 4.95, 3.6, 0.65, "Load Custom YOLO Object & Currency Models\nInitialize EasyOCR, Camera & HC-SR04 (10 Hz)", bg="#dbeafe", edge="#1d4ed8")
    _draw_box(ax, 3.2, 4.0, 3.6, 0.65, "Capture Real-Time Video Frame (30 FPS)\n& Poll Ultrasonic Distance Sensor", bg="#e0e7ff", edge="#4338ca")

    # 4 Parallel branches
    _draw_box(ax, 0.3, 2.65, 2.1, 0.95, "Ultrasonic Branch\nCheck Distance <= 2m\nMoving Avg Filter\n(Priority 1)", bg="#fee2e2", edge="#b91c1c", fontsize=8.5)
    _draw_box(ax, 2.7, 2.65, 2.1, 0.95, "Object Detection\nYOLOv8n / MobileNet\nNMS + IoU Tracking\n(Priority 2)", bg="#fae8ff", edge="#a21caf", fontsize=8.5)
    _draw_box(ax, 5.1, 2.65, 2.1, 0.95, "Currency Detection\nCustom INR YOLO\n₹10..₹500 Classification\n(Priority 3)", bg="#dcfce7", edge="#15803d", fontsize=8.5)
    _draw_box(ax, 7.5, 2.65, 2.1, 0.95, "OCR Text Recognition\nEasyOCR (CRAFT+CRNN)\nSignboard Reading\n(Priority 4)", bg="#ffedd5", edge="#c2410c", fontsize=8.5)

    _draw_box(ax, 2.5, 1.4, 5.0, 0.75, "Priority Alert Arbitration & Cooldown Deduplication\nGenerate Directional Voice Guidance (Left / Ahead / Right)", bg="#fef3c7", edge="#b45309")
    _draw_box(ax, 2.5, 0.35, 5.0, 0.7, "Dispatch Non-Blocking Audio via pyttsx3 to Earphones\n& Update Live Streamlit HUD -> Repeat Next Cycle", bg="#dbeafe", edge="#1d4ed8")

    _draw_arrow(ax, 5.0, 5.9, 5.0, 5.6)
    _draw_arrow(ax, 5.0, 4.95, 5.0, 4.65)
    for bx in (1.35, 3.75, 6.15, 8.55):
        _draw_arrow(ax, 5.0, 4.0, bx, 3.6)
        _draw_arrow(ax, bx, 2.65, 5.0, 2.15)
    _draw_arrow(ax, 5.0, 1.4, 5.0, 1.05)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_fig_6_2_hardware_wiring(out_path: Path) -> None:
    """Generate Fig. 6.2: Raspberry Pi 4 CSI Camera, HC-SR04 GPIO23/24 Divider & Audio Wiring."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        5.0,
        4.85,
        "Fig. 6.2: Raspberry Pi 4 Hardware Interfacing (CSI Camera, HC-SR04 Voltage Divider & Audio)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
    )

    _draw_box(ax, 3.4, 1.1, 3.2, 3.2, "Raspberry Pi 4 Model B\n(4GB RAM)\n\n• Pin 2: 5V Power\n• Pin 6: GND\n• Pin 16: BCM GPIO 23 (OUT)\n• Pin 18: BCM GPIO 24 (IN 3.3V)\n• CSI Ribbon Port\n• 3.5mm / USB Audio Port", bg="#dcfce7", edge="#15803d")
    _draw_box(ax, 0.3, 2.9, 2.3, 1.2, "5-MP Pi Camera\nModule\n(CSI Ribbon Cable)", bg="#dbeafe", edge="#1d4ed8")
    _draw_box(ax, 0.3, 1.0, 2.3, 1.2, "Earphones / Speakers\n& 5V 10,000mAh\nPower Bank", bg="#f3e8ff", edge="#6d28d9")
    _draw_box(ax, 7.3, 1.6, 2.4, 2.2, "HC-SR04 Ultrasonic\nDistance Sensor\n\n• VCC (5V)\n• TRIG (Input)\n• ECHO (5V Output via\n  1kΩ / 2kΩ Divider)\n• GND", bg="#fef3c7", edge="#b45309")

    _draw_arrow(ax, 2.6, 3.5, 3.4, 3.5)
    _draw_arrow(ax, 3.4, 1.6, 2.6, 1.6)
    _draw_arrow(ax, 6.6, 3.1, 7.3, 3.1)
    _draw_arrow(ax, 7.3, 2.2, 6.6, 2.2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    graphs_dir = Path(__file__).resolve().parent.parent / "evaluation" / "graphs"
    generate_fig_5_1_block_diagram(graphs_dir / "blackbook_fig_5_1_architecture.png")
    generate_fig_6_1_workflow_diagram(graphs_dir / "blackbook_fig_6_1_workflow.png")
    generate_fig_6_2_hardware_wiring(graphs_dir / "blackbook_fig_6_2_wiring.png")
    print("Generated Blackbook architecture, workflow, and wiring figures.")

import argparse
import csv
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def y_center(self) -> float:
        return 0.5 * (self.y_min + self.y_max)

    @property
    def y_top(self) -> float:
        return self.y_min

    @property
    def y_bottom(self) -> float:
        return self.y_max


def _parse_matrix(transform: str) -> tuple[float, float, float, float, float, float]:
    m = re.search(r"matrix\(([^)]+)\)", transform or "")
    if not m:
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != 6:
        raise ValueError(f"Unexpected transform matrix: {transform}")
    a, b, c, d, e, f = (float(p) for p in parts)
    return (a, b, c, d, e, f)


def _apply_matrix(x: float, y: float, m: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_rect_from_path_d(d: str) -> Rect | None:
    if not d:
        return None
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", d)]
    if len(nums) < 8:
        return None
    points = list(zip(nums[0::2], nums[1::2]))
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    return Rect(min(x_vals), max(x_vals), min(y_vals), max(y_vals))


def _find_axis_bounds(svg_text: str) -> tuple[float, float]:
    # Look for the y-axis line in the *pre-transform* coordinate system and apply its matrix.
    # This is stable for the source figure.
    pattern = (
        r'<path style="[^"]*stroke-width:7[^"]*" '
        r'd="M 355\.99373 2214\.937564 L 355\.99373 90\.001547 " '
        r'transform="([^"]+)"'
    )
    m = re.search(pattern, svg_text)
    if not m:
        raise RuntimeError("Failed to locate axis line for y-bounds in SVG.")
    matrix = _parse_matrix(m.group(1))
    _, y_bottom = _apply_matrix(355.99373, 2214.937564, matrix)
    _, y_top = _apply_matrix(355.99373, 90.001547, matrix)
    return (y_top, y_bottom)


def _extract_method_colors_in_order(svg_text: str) -> list[str]:
    # Capture unique fill colors in order of first appearance.
    seen: list[str] = []
    for m in re.finditer(r"fill:rgb\(([^)]*)\)", svg_text):
        c = m.group(1)
        if c in ("0%,0%,0%", "100%,100%,100%"):
            continue
        if c not in seen:
            seen.append(c)
    return seen


def _parse_bbox_words(bbox_xml_path: str) -> list[tuple[str, float, float, float, float]]:
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    tree = ET.parse(bbox_xml_path)
    root = tree.getroot()
    words = []
    for w in root.findall(".//x:word", ns):
        t = (w.text or "").strip()
        if not t:
            continue
        words.append(
            (
                t,
                float(w.attrib["xMin"]),
                float(w.attrib["yMin"]),
                float(w.attrib["xMax"]),
                float(w.attrib["yMax"]),
            )
        )
    return words


def _cluster_words_into_lines(
    words: list[tuple[str, float, float, float, float]], y_tol: float = 2.0
) -> list[tuple[str, Rect]]:
    # Group words by yMin (legend lines are horizontally aligned).
    words_sorted = sorted(words, key=lambda x: (x[2], x[1]))
    lines: list[list[tuple[str, float, float, float, float]]] = []
    for w in words_sorted:
        if not lines:
            lines.append([w])
            continue
        last = lines[-1]
        if abs(w[2] - last[0][2]) <= y_tol:
            last.append(w)
        else:
            lines.append([w])

    out: list[tuple[str, Rect]] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda x: x[1])
        text = "".join(w[0] for w in line_sorted)
        x_min = min(w[1] for w in line_sorted)
        y_min = min(w[2] for w in line_sorted)
        x_max = max(w[3] for w in line_sorted)
        y_max = max(w[4] for w in line_sorted)
        out.append((text, Rect(x_min, x_max, y_min, y_max)))
    return out


def _find_legend_label_bboxes(bbox_xml_path: str, targets: list[str]) -> dict[str, Rect]:
    words = _parse_bbox_words(bbox_xml_path)
    lines = _cluster_words_into_lines(words)

    # Clean texts for robust matching (e.g., "DualRTMO" vs "Dual RTMO").
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    target_norm = {t: norm(t) for t in targets}
    found: dict[str, Rect] = {}
    for text, rect in lines:
        n = norm(text)
        for t, tn in target_norm.items():
            if t in found:
                continue
            if n == tn or tn in n:
                found[t] = rect
    missing = [t for t in targets if t not in found]
    if missing:
        raise RuntimeError(f"Failed to find legend labels in bbox XML: {missing}")
    return found


def _pick_legend_boxes(rects_by_color: dict[str, list[Rect]], y_bottom: float) -> list[tuple[str, Rect]]:
    # Legend boxes do not touch the plot bottom; keep small rectangles.
    legend: list[tuple[str, Rect]] = []
    for color, rects in rects_by_color.items():
        if color in ("0%,0%,0%", "100%,100%,100%"):
            continue
        candidates = [r for r in rects if abs(r.y_bottom - y_bottom) >= 1e-2]
        if not candidates:
            continue
        # Choose the smallest-area rectangle as the legend box.
        candidates.sort(key=lambda r: (r.x_max - r.x_min) * (r.y_max - r.y_min))
        legend.append((color, candidates[0]))
    return legend


def _match_labels_to_colors(
    label_bboxes: dict[str, Rect], legend_boxes: list[tuple[str, Rect]]
) -> dict[str, str]:
    # Match each label to the nearest legend box on its left (same y band).
    mapping: dict[str, str] = {}
    for label, rect in label_bboxes.items():
        best = None
        best_d = float("inf")
        for color, box in legend_boxes:
            # Prefer boxes left of the text.
            if box.x_center > rect.x_min:
                continue
            dy = abs(box.y_center - rect.y_center)
            dx = rect.x_min - box.x_center
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best = color
        if best is None:
            raise RuntimeError(f"Failed to match legend box for label: {label}")
        mapping[label] = best
    return mapping


def _parse_svg_rects(svg_path: str) -> dict[str, list[Rect]]:
    ns = {"svg": "http://www.w3.org/2000/svg"}
    tree = ET.parse(svg_path)
    root = tree.getroot()

    rects_by_color: dict[str, list[Rect]] = {}

    for path in root.findall(".//svg:path", ns):
        style = path.attrib.get("style", "")
        if "fill:rgb(" not in style:
            continue
        m_color = re.search(r"fill:rgb\(([^)]*)\)", style)
        if not m_color:
            continue
        color = m_color.group(1)
        if color in ("0%,0%,0%", "100%,100%,100%"):
            continue

        d = path.attrib.get("d", "")
        rect = _parse_rect_from_path_d(d)
        if rect is None:
            continue

        transform = path.attrib.get("transform")
        if transform:
            matrix = _parse_matrix(transform)
            corners = [
                (rect.x_min, rect.y_min),
                (rect.x_min, rect.y_max),
                (rect.x_max, rect.y_min),
                (rect.x_max, rect.y_max),
            ]
            corners_t = [_apply_matrix(x, y, matrix) for x, y in corners]
            xs = [p[0] for p in corners_t]
            ys = [p[1] for p in corners_t]
            rect = Rect(min(xs), max(xs), min(ys), max(ys))

        rects_by_color.setdefault(color, []).append(rect)

    return rects_by_color


def _bars_to_series(
    rects: list[Rect], y_bottom: float, y_top: float, y_axis_max: float = 3500.0
) -> list[float]:
    # Keep only bars that touch the plot bottom (legend boxes do not).
    bars = [r for r in rects if abs(r.y_bottom - y_bottom) < 1e-2]
    bars.sort(key=lambda r: r.x_center)
    if len(bars) != 10:
        raise RuntimeError(f"Expected 10 bars (people=1..10), got {len(bars)}")

    scale = (y_bottom - y_top) / y_axis_max
    if scale <= 0:
        raise RuntimeError("Invalid y-axis scale.")

    series = []
    for r in bars:
        value = (y_bottom - r.y_top) / scale
        series.append(value)
    return series


def _plot_pdf(out_pdf: str, people: list[int], series: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {
        "本文方法": "#1f77b4",
        "RTMW3D（单目）": "#d62728",
        "RTMO+MobileStereoNet（级联）": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)

    ax_left, ax_right = axes
    ax_left.plot(
        people,
        series["RTMW3D（单目）"],
        marker="o",
        linewidth=1.9,
        markersize=3.8,
        color=colors["RTMW3D（单目）"],
        label="RTMW3D（单目）",
    )
    ax_left.set_title("(a) 自顶向下方法的时延增长", fontsize=11)
    ax_left.set_xlabel("人数")
    ax_left.set_ylabel("推理时延（ms）")
    ax_left.set_xticks(people)
    ax_left.set_ylim(350, 2500)
    ax_left.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_left.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    for label in ("本文方法", "RTMO+MobileStereoNet（级联）"):
        ax_right.plot(
            people,
            series[label],
            marker="o",
            linewidth=1.9,
            markersize=3.8,
            color=colors[label],
            label=label,
        )
    ax_right.set_title("(b) 低时延且人数无关的对比", fontsize=11)
    ax_right.set_xlabel("人数")
    ax_right.set_ylabel("推理时延（ms）")
    ax_right.set_xticks(people)
    ax_right.set_ylim(110, 205)
    ax_right.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_right.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=7.6,
        loc="center",
        bbox_to_anchor=(0.66, 0.43),
        ncol=1,
        borderaxespad=0.0,
    )

    fig.tight_layout(w_pad=0.9, rect=(0.0, 0.02, 1.0, 0.98))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)


def _load_series_from_csv(csv_path: str) -> tuple[list[int], dict[str, list[float]]]:
    people: list[int] = []
    ours: list[float] = []
    rtmw3d: list[float] = []
    rtmo_msn: list[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people.append(int(row["people"]))
            ours.append(float(row["ours_ms"]))
            rtmw3d.append(float(row["rtmw3d_ms"]))
            rtmo_msn.append(float(row["rtmo_msn_ms"]))
    return people, {
        "本文方法": ours,
        "RTMW3D（单目）": rtmw3d,
        "RTMO+MobileStereoNet（级联）": rtmo_msn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="StereoMulti3DPose/figures/latency.pdf")
    ap.add_argument("--work-svg", default="build/latency_tmp.svg")
    ap.add_argument("--work-bbox", default="build/latency_bbox.xml")
    ap.add_argument("--csv-out", default="tools/latency_digitize/latency_vs_people.csv")
    ap.add_argument("--pdf-out", default="images/latency_vs_people.pdf")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        people, keep = _load_series_from_csv(args.csv_out)
        _plot_pdf(args.pdf_out, people, keep)
        return 0

    os.makedirs(os.path.dirname(args.work_svg), exist_ok=True)
    subprocess.check_call(["pdftocairo", "-svg", args.input, args.work_svg])
    subprocess.check_call(["pdftotext", "-bbox", args.input, args.work_bbox])

    svg_text = open(args.work_svg, "r", encoding="utf-8", errors="ignore").read()
    y_top, y_bottom = _find_axis_bounds(svg_text)

    rects_by_color = _parse_svg_rects(args.work_svg)
    legend_boxes = _pick_legend_boxes(rects_by_color, y_bottom=y_bottom)

    # Extract legend text locations from the PDF, then match to legend boxes to get colors.
    needed = ["Ours", "RTMW3D", "RTMO+MobileStereoNet"]
    label_bboxes = _find_legend_label_bboxes(args.work_bbox, needed)
    label_to_color = _match_labels_to_colors(label_bboxes, legend_boxes)

    series_all: dict[str, list[float]] = {}
    for label in needed:
        color = label_to_color[label]
        rects = rects_by_color.get(color, [])
        series_all[label] = _bars_to_series(rects, y_bottom=y_bottom, y_top=y_top)

    people = list(range(1, 11))
    keep = {
        "本文方法": series_all["Ours"],
        "RTMW3D（单目）": series_all["RTMW3D"],
        "RTMO+MobileStereoNet（级联）": series_all["RTMO+MobileStereoNet"],
    }

    # Snap the key points of "本文方法" to the values explicitly reported in the source paper
    # (Xiaomi 14: 116.0 ms for 1 person, 117.3 ms for 10 people), to keep the thesis text,
    # table, and figure consistent.
    ours = keep["本文方法"]
    if len(ours) == 10 and abs(ours[0] - ours[-1]) > 1e-6:
        target_1, target_10 = 116.0, 117.3
        a = (target_10 - target_1) / (ours[-1] - ours[0])
        b = target_1 - a * ours[0]
        keep["本文方法"] = [a * v + b for v in ours]

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["people", "ours_ms", "rtmw3d_ms", "rtmo_msn_ms"])
        for i, p in enumerate(people):
            w.writerow([p, keep["本文方法"][i], keep["RTMW3D（单目）"][i], keep["RTMO+MobileStereoNet（级联）"][i]])

    _plot_pdf(args.pdf_out, people, keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

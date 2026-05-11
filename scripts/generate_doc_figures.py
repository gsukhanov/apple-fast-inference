from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO_ROOT / "build" / "fastinf_benchmark"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "static" / "generated"

TENSOR_MUL_RE = re.compile(
    r"BM_TensorMul<\s*DType::(?P<dtype>[^,>]+),\s*DeviceLikeType::(?P<device>[^>]+)>"
    r"/(?P<m>\d+)/(?P<k>\d+)/(?P<n>\d+)"
)
LENET_RE = re.compile(
    r"BM_LeNet(?P<dtype>Float32|Float64)(?P<device>Neon|CPU|AMX)/(?P<batch>\d+)"
)


@dataclass(frozen=True)
class BenchmarkPoint:
    name: str
    dtype: str
    device: str
    m: int
    k: int
    n: int
    real_time_ns: float
    cpu_time_ns: float
    items_per_second: float | None
    label: str | None = None

    @property
    def shape(self) -> str:
        if self.label is not None:
            return self.label
        return f"{self.m}x{self.k} * {self.k}x{self.n}"

    @property
    def latency_ms(self) -> float:
        return self.real_time_ns / 1_000_000.0

    @property
    def gops(self) -> float:
        if self.items_per_second is None:
            return 0.0
        return self.items_per_second / 1_000_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CPU/NEON and AMX performance charts for Typst documentation."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"Path to benchmark executable. Default: {DEFAULT_BENCHMARK}",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="Read an existing Google Benchmark JSON file instead of running a binary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--benchmark-filter",
        default="TensorMul",
        help="Google Benchmark filter used when running the benchmark binary.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        default="0.01s",
        help="Google Benchmark minimum time used for documentation runs.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not run the benchmark binary. Requires --benchmark-json.",
    )
    return parser.parse_args()


def run_benchmark(binary: Path, benchmark_filter: str, min_time: str) -> dict:
    if not binary.exists():
        raise FileNotFoundError(
            f"Benchmark binary not found: {binary}. Build the project first or pass --benchmark-json."
        )

    command = [
        str(binary),
        "--benchmark_format=json",
        f"--benchmark_filter={benchmark_filter}",
        f"--benchmark_min_time={min_time}",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")

    return json.loads(completed.stdout)


def read_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def extract_points(results: dict) -> list[BenchmarkPoint]:
    points: list[BenchmarkPoint] = []
    for bench in results.get("benchmarks", []):
        if bench.get("run_type") not in (None, "iteration"):
            continue

        name = bench.get("run_name") or bench.get("name") or ""
        tensor_mul_match = TENSOR_MUL_RE.search(name)
        lenet_match = LENET_RE.search(name)
        if tensor_mul_match:
            points.append(
                BenchmarkPoint(
                    name=name,
                    dtype=tensor_mul_match.group("dtype"),
                    device=tensor_mul_match.group("device"),
                    m=int(tensor_mul_match.group("m")),
                    k=int(tensor_mul_match.group("k")),
                    n=int(tensor_mul_match.group("n")),
                    real_time_ns=float(bench["real_time"]),
                    cpu_time_ns=float(bench["cpu_time"]),
                    items_per_second=(
                        float(bench["items_per_second"])
                        if "items_per_second" in bench
                        else None
                    ),
                )
            )
            continue

        if lenet_match:
            dtype = {
                "Float32": "float32",
                "Float64": "float64",
            }[lenet_match.group("dtype")]
            device = {
                "CPU": "cpu",
                "Neon": "neon",
                "AMX": "amx",
            }[lenet_match.group("device")]
            batch = int(lenet_match.group("batch"))
            points.append(
                BenchmarkPoint(
                    name=name,
                    dtype=dtype,
                    device=device,
                    m=batch,
                    k=1,
                    n=28 * 28,
                    real_time_ns=float(bench["real_time"]),
                    cpu_time_ns=float(bench["cpu_time"]),
                    items_per_second=(
                        float(bench["items_per_second"])
                        if "items_per_second" in bench
                        else None
                    ),
                    label=f"batch={batch}",
                )
            )

    if not points:
        raise ValueError("No TensorMul or LeNet benchmark results were found in the JSON data.")

    return sorted(points, key=lambda p: (p.dtype, p.m, p.k, p.n, p.device))


def write_raw_json(results: dict, output_dir: Path) -> Path:
    output_path = output_dir / "benchmark_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path


def write_csv(points: Iterable[BenchmarkPoint], output_dir: Path) -> Path:
    output_path = output_dir / "performance_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "dtype",
                "device",
                "shape",
                "m",
                "k",
                "n",
                "real_time_ns",
                "latency_ms",
                "items_per_second",
                "gops",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.dtype,
                    point.device,
                    point.shape,
                    point.m,
                    point.k,
                    point.n,
                    f"{point.real_time_ns:.6f}",
                    f"{point.latency_ms:.6f}",
                    "" if point.items_per_second is None else f"{point.items_per_second:.6f}",
                    f"{point.gops:.6f}",
                ]
            )
    return output_path


def write_typst_snippet(output_dir: Path) -> Path:
    output_path = output_dir / "README.typ"
    content = """// Generated documentation figures.
// Add the needed lines to a Typst template section.

#figure(
  image("../../static/generated/cpu_amx_latency.svg", width: 100%),
  caption: [Сравнение времени выполнения CPU Conv2d, NEON im2col и AMX Conv2d],
)

#figure(
  image("../../static/generated/cpu_amx_matmul_latency.svg", width: 100%),
  caption: [Сравнение времени выполнения матричного умножения CPU/NEON и AMX],
)
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def grouped_labels(points: list[BenchmarkPoint]) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for point in sorted(points, key=lambda p: (p.dtype, p.m, p.k, p.n)):
        key = (point.dtype, point.shape)
        if key in seen:
            continue
        seen.add(key)
        labels.append(key)
    return labels


def device_value_map(points: list[BenchmarkPoint], metric: str) -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for point in points:
        key = (point.dtype, point.shape)
        value = getattr(point, metric)
        values.setdefault(key, {})[point.device] = 0.0 if value is None else value
    return values


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def log_bounds(values: Iterable[float]) -> tuple[int, int]:
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return 0, 1

    min_power = math.floor(math.log10(min(positive_values)))
    max_power = math.ceil(math.log10(max(positive_values)))
    if min_power == max_power:
        max_power += 1
    return min_power, max_power


def format_power_tick(power: int) -> str:
    value = 10**power
    if -3 <= power <= 4:
        return f"{value:g}"
    return f"1e{power}"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_lines(text: str) -> list[str]:
    return text.split("\\n")


def svg_text(x: float, y: float, text: str, *, anchor: str = "middle", size: int = 12) -> str:
    lines = text_lines(text)
    if len(lines) == 1:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="#1f2937">{svg_escape(text)}</text>'
        )

    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size + 2
        tspans.append(
            f'<tspan x="{x:.2f}" dy="{dy}">{svg_escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="#1f2937">' + "".join(tspans) + "</text>"
    )


def render_grouped_bar_chart(
    output_path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    y_scale: str = "log",
) -> None:
    width = 1180
    height = 720
    margin_left = 92
    margin_right = 34
    margin_top = 82
    margin_bottom = 150
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    all_values = [value for _, values, _ in series for value in values]
    max_value = max(all_values, default=1.0)
    y_max = nice_max(max_value)
    log_min_power, log_max_power = log_bounds(all_values)
    log_span = max(log_max_power - log_min_power, 1)
    group_count = max(len(labels), 1)
    group_width = chart_width / group_count
    bar_gap = 8
    bar_width = min(46, (group_width - 34) / max(len(series), 1) - bar_gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.2f}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="22" y="{margin_top + chart_height / 2:.2f}" text-anchor="middle" font-size="14" fill="#374151" transform="rotate(-90 22 {margin_top + chart_height / 2:.2f})">{svg_escape(y_label)}</text>',
    ]

    if y_scale == "log":
        tick_powers = list(range(log_min_power, log_max_power + 1))
        for power in tick_powers:
            y = margin_top + chart_height - chart_height * (power - log_min_power) / log_span
            parts.append(
                f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{format_power_tick(power)}</text>'
            )
    else:
        for tick in range(6):
            value = y_max * tick / 5
            y = margin_top + chart_height - chart_height * value / y_max
            parts.append(
                f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{value:.2g}</text>'
            )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#111827" stroke-width="1.2"/>'
    )

    legend_x = margin_left
    for index, (name, _, color) in enumerate(series):
        x = legend_x + index * 132
        parts.append(f'<rect x="{x}" y="54" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 22}" y="67" font-size="13" fill="#374151">{svg_escape(name)}</text>'
        )

    for group_index, label in enumerate(labels):
        center_x = margin_left + group_width * group_index + group_width / 2
        first_bar_x = center_x - (len(series) * bar_width + (len(series) - 1) * bar_gap) / 2

        for series_index, (_, values, color) in enumerate(series):
            value = values[group_index]
            if y_scale == "log":
                if value <= 0:
                    bar_height = 0
                else:
                    value_power = max(math.log10(value), log_min_power)
                    bar_height = chart_height * (value_power - log_min_power) / log_span
            else:
                bar_height = chart_height * value / y_max if y_max else 0
            x = first_bar_x + series_index * (bar_width + bar_gap)
            y = margin_top + chart_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="4" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="10" fill="#374151">{value:.2g}</text>'
            )

        parts.append(svg_text(center_x, margin_top + chart_height + 24, label, size=11))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_single_bar_chart(
    output_path: Path,
    title: str,
    y_label: str,
    values: list[tuple[str, float]],
    color: str,
) -> None:
    labels = [label for label, _ in values]
    series_values = [value for _, value in values]
    render_grouped_bar_chart(
        output_path, title, y_label, labels, [("AMX / CPU", series_values, color)]
    )


def make_latency_series(
    points: list[BenchmarkPoint],
    devices: list[str],
    names: dict[str, str],
    colors: dict[str, str],
) -> list[tuple[str, list[float], str]]:
    latency_by_device = device_value_map(points, "latency_ms")
    return [
        (
            names.get(device, device),
            [
                latency_by_device.get((dtype, shape), {}).get(device, 0.0)
                for dtype, shape in grouped_labels(points)
            ],
            colors.get(device, "#4b5563"),
        )
        for device in devices
    ]


def generate_lenet_latency_chart(points: list[BenchmarkPoint], output_dir: Path) -> Path:
    labels = [f"{dtype}\\n{shape}" for dtype, shape in grouped_labels(points)]
    path = output_dir / "cpu_amx_latency.svg"
    render_grouped_bar_chart(
        path,
        "Время выполнения инференса LeNet",
        "Время, мс (логарифмическая шкала)",
        labels,
        make_latency_series(
            points,
            ["cpu", "neon", "amx"],
            {
                "cpu": "CPU Conv2d",
                "neon": "NEON im2col",
                "amx": "AMX Conv2d",
            },
            {
                "cpu": "#2563eb",
                "neon": "#f59e0b",
                "amx": "#dc2626",
            },
        ),
    )
    return path


def generate_matmul_latency_chart(points: list[BenchmarkPoint], output_dir: Path) -> Path:
    labels = [f"{dtype}\\n{shape}" for dtype, shape in grouped_labels(points)]
    path = output_dir / "cpu_amx_matmul_latency.svg"
    render_grouped_bar_chart(
        path,
        "Время выполнения матричного умножения CPU/NEON и AMX",
        "Время, мс (логарифмическая шкала)",
        labels,
        make_latency_series(
            points,
            ["cpu", "amx"],
            {
                "cpu": "CPU/NEON",
                "amx": "AMX",
            },
            {
                "cpu": "#2563eb",
                "amx": "#dc2626",
            },
        ),
    )
    return path


def generate_charts(points: list[BenchmarkPoint], output_dir: Path) -> list[Path]:
    lenet_points = [point for point in points if point.label is not None]
    matmul_points = [point for point in points if point.label is None]

    generated: list[Path] = []
    if lenet_points:
        generated.append(generate_lenet_latency_chart(lenet_points, output_dir))
    if matmul_points:
        generated.append(generate_matmul_latency_chart(matmul_points, output_dir))
    return generated


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.benchmark_json:
            results = read_results(args.benchmark_json)
        elif args.no_run:
            raise ValueError("--no-run requires --benchmark-json.")
        else:
            results = run_benchmark(
                args.benchmark.resolve(), args.benchmark_filter, args.benchmark_min_time
            )

        points = extract_points(results)
        generated = [
            write_raw_json(results, output_dir),
            write_csv(points, output_dir),
            write_typst_snippet(output_dir),
            *generate_charts(points, output_dir),
        ]
    except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("Generated documentation artifacts:")
    for path in generated:
        print(f"  {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

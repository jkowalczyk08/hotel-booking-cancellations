from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence


_MIME_PRIORITY: tuple[tuple[str, str], ...] = (
    ("image/png", ".png"),
    ("image/jpeg", ".jpg"),
    ("image/svg+xml", ".svg"),
)


@dataclass(frozen=True)
class SavedPlot:
    notebook_path: Path
    cell_index: int
    output_index: int
    mime_type: str
    image_path: Path


def _group_plots_by_output_number(plots_by_notebook: dict[Path, list[SavedPlot]]) -> dict[int, list[SavedPlot]]:
    grouped: dict[int, list[SavedPlot]] = {}
    for notebook_plots in plots_by_notebook.values():
        for plot in notebook_plots:
            output_number = plot.output_index + 1
            grouped.setdefault(output_number, []).append(plot)

    for output_number, plots in grouped.items():
        grouped[output_number] = sorted(
            plots,
            key=lambda plot: (plot.notebook_path.name.lower(), plot.cell_index, plot.output_index),
        )

    return grouped


def _normalize_payload(raw_payload: Any) -> str:
    if isinstance(raw_payload, list):
        return "".join(str(part) for part in raw_payload)
    return str(raw_payload)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower() or "notebook"


def _discover_target_notebooks(notebooks_dir: Path, name_filters: Sequence[str]) -> list[Path]:
    lowered_filters = tuple(token.lower() for token in name_filters)
    candidates = sorted(notebooks_dir.rglob("*.ipynb"))
    return [path for path in candidates if any(token in path.name.lower() for token in lowered_filters)]


def extract_plots_from_notebook(notebook_path: Path, images_root: Path) -> list[SavedPlot]:
    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook_json.get("cells", [])

    notebook_output_dir = images_root / _slugify(notebook_path.stem)
    notebook_output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[SavedPlot] = []

    for cell_index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        for output_index, output in enumerate(outputs):
            data = output.get("data")
            if not isinstance(data, dict):
                continue

            selected_mime: str | None = None
            selected_extension: str | None = None
            for mime_type, extension in _MIME_PRIORITY:
                if mime_type in data:
                    selected_mime = mime_type
                    selected_extension = extension
                    break

            if selected_mime is None or selected_extension is None:
                continue

            payload = _normalize_payload(data[selected_mime])
            image_path = (
                notebook_output_dir
                / f"{_slugify(notebook_path.stem)}_cell_{cell_index + 1:03d}_out_{output_index + 1:03d}{selected_extension}"
            )

            if selected_mime in {"image/png", "image/jpeg"}:
                image_path.write_bytes(base64.b64decode(payload))
            else:
                image_path.write_text(payload, encoding="utf-8")

            extracted.append(
                SavedPlot(
                    notebook_path=notebook_path,
                    cell_index=cell_index,
                    output_index=output_index,
                    mime_type=selected_mime,
                    image_path=image_path,
                )
            )

    return extracted


def write_markdown_report(plots_by_notebook: dict[Path, list[SavedPlot]], markdown_path: Path) -> Path:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_by_output = _group_plots_by_output_number(plots_by_notebook)

    lines: list[str] = [
        "# Extracted Notebook Plots",
        "",
        f"_Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_",
        "",
    ]

    if not grouped_by_output:
        lines.append("No image outputs found in matching notebooks.")
        lines.append("")
    else:
        for output_number in sorted(grouped_by_output):
            lines.append(f"## Output {output_number}")
            lines.append("")

            for plot in grouped_by_output[output_number]:
                relative_image_path = os.path.relpath(plot.image_path, start=markdown_path.parent).replace("\\", "/")
                lines.append(f"### `{plot.notebook_path.name}` - Cell {plot.cell_index + 1}")
                lines.append("")
                lines.append(f"![{plot.notebook_path.stem} cell {plot.cell_index + 1}]({relative_image_path})")
                lines.append("")

    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return markdown_path


def _to_notebook_source(markdown: str) -> list[str]:
    if not markdown.endswith("\n"):
        markdown = f"{markdown}\n"
    return markdown.splitlines(keepends=True)


def write_jupyter_report(plots_by_notebook: dict[Path, list[SavedPlot]], report_path: Path) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    grouped_by_output = _group_plots_by_output_number(plots_by_notebook)

    cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": _to_notebook_source(f"# Extracted Notebook Plots\n\nGenerated on {generated_at} UTC"),
        }
    ]

    if not grouped_by_output:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _to_notebook_source("No image outputs found in matching notebooks."),
            }
        )
    else:
        for output_number in sorted(grouped_by_output):
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _to_notebook_source(f"## Output {output_number}"),
                }
            )

            for plot in grouped_by_output[output_number]:
                relative_image_path = os.path.relpath(plot.image_path, start=report_path.parent).replace("\\", "/")
                markdown_block = "\n".join(
                    [
                        f"### `{plot.notebook_path.name}` - Cell {plot.cell_index + 1}",
                        "",
                        f"![{plot.notebook_path.stem} cell {plot.cell_index + 1}]({relative_image_path})",
                    ]
                )
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": _to_notebook_source(markdown_block),
                    }
                )

    notebook_payload: dict[str, Any] = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    report_path.write_text(json.dumps(notebook_payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def extract_filtered_notebook_plots(
    notebooks_dir: Path,
    output_dir: Path,
    name_filters: Sequence[str] = ("enhanced", "no_leakage"),
    report_filename: str = "plots.ipynb",
    report_format: Literal["md", "ipynb"] = "ipynb",
) -> tuple[Path, dict[Path, list[SavedPlot]]]:
    images_root = output_dir / "images"
    target_notebooks = _discover_target_notebooks(notebooks_dir=notebooks_dir, name_filters=name_filters)

    plots_by_notebook: dict[Path, list[SavedPlot]] = {}
    for notebook_path in target_notebooks:
        plots_by_notebook[notebook_path] = extract_plots_from_notebook(
            notebook_path=notebook_path,
            images_root=images_root,
        )

    report_path = output_dir / report_filename
    if report_format == "md":
        report_path = write_markdown_report(
            plots_by_notebook=plots_by_notebook,
            markdown_path=report_path,
        )
    elif report_format == "ipynb":
        report_path = write_jupyter_report(
            plots_by_notebook=plots_by_notebook,
            report_path=report_path,
        )
    else:
        raise ValueError(f"Unsupported report format: {report_format}")

    return report_path, plots_by_notebook


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract plots from selected Jupyter notebooks.")
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path("notebooks"),
        help="Directory to scan for .ipynb files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/notebook_plots"),
        help="Directory where extracted images and report are saved.",
    )
    parser.add_argument(
        "--report-format",
        choices=("md", "ipynb"),
        default="ipynb",
        help="Report format to generate.",
    )
    parser.add_argument(
        "--report-filename",
        default=None,
        help="Output report filename (inside --output-dir). Defaults to plots.<report-format>.",
    )
    parser.add_argument(
        "--name-filter",
        nargs="+",
        default=("enhanced", "no_leakage"),
        help="Notebook name substrings to include.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report_filename = args.report_filename or f"plots.{args.report_format}"

    report_path, plots_by_notebook = extract_filtered_notebook_plots(
        notebooks_dir=args.notebooks_dir,
        output_dir=args.output_dir,
        name_filters=args.name_filter,
        report_filename=report_filename,
        report_format=args.report_format,
    )

    notebook_count = len(plots_by_notebook)
    plot_count = sum(len(plots) for plots in plots_by_notebook.values())

    print(f"Scanned notebooks: {notebook_count}")
    print(f"Extracted plots: {plot_count}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

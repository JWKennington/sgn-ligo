"""Generate a self-contained HTML report from a coinc XML file.

This script reads a GraceDB-compatible coinc XML file and produces a
standalone HTML page with:
- Event summary table
- Per-detector trigger tables
- PSD plots (embedded as base64 PNG)
- SNR time series plots (embedded as base64 PNG)

Usage:
    sgn-ligo-coinc-to-html coinc.xml [output.html]

If output.html is not specified, it will be named based on the input file.
"""

# Copyright (C) 2025 Chad Hanna

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict

import lal.series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils

# Use non-interactive backend for headless operation
matplotlib.use("Agg")


def _parse_coinc_xml(filepath: str) -> Dict:
    """Parse a coinc XML file and extract all relevant data.

    Args:
        filepath: Path to the coinc XML file

    Returns:
        Dictionary with event data, triggers, PSDs, and SNR time series
    """
    xmldoc = ligolw_utils.load_filename(filepath)

    # Extract coinc inspiral (event summary)
    coinc_inspiral = lsctables.CoincInspiralTable.get_table(xmldoc)[0]

    # Extract single inspiral triggers
    sngl_inspirals = list(lsctables.SnglInspiralTable.get_table(xmldoc))

    # Extract PSDs
    psds = lal.series.read_psd_xmldoc(xmldoc)

    # Extract SNR time series
    snr_series = {}
    root = xmldoc.childNodes[0]
    for child in root.childNodes:
        if hasattr(child, "tagName") and child.tagName == "LIGO_LW":
            try:
                ts = lal.series.parse_COMPLEX8TimeSeries(child)
                # Extract IFO from time series name (format: "snr_H1", etc.)
                if ts.name and ts.name.startswith("snr_"):
                    ifo = ts.name[4:]  # Strip "snr_" prefix
                    snr_series[ifo] = ts
            except Exception:  # noqa: S110, S112
                continue

    return {
        "coinc_inspiral": coinc_inspiral,
        "triggers": sngl_inspirals,
        "psds": psds,
        "snr_series": snr_series,
    }


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_psd(psd: lal.REAL8FrequencySeries, ifo: str) -> str:
    """Create a PSD plot and return as base64 PNG.

    Args:
        psd: LAL frequency series containing the PSD
        ifo: Detector name for the title

    Returns:
        Base64-encoded PNG string
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    freqs = psd.f0 + np.arange(psd.data.length) * psd.deltaF
    asd = np.sqrt(psd.data.data)

    ax.loglog(freqs[1:], asd[1:], color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("ASD (strain/√Hz)", fontsize=12)
    ax.set_title(f"{ifo} Amplitude Spectral Density", fontsize=14, fontweight="bold")
    ax.set_xlim(10, freqs[-1])
    ax.grid(True, alpha=0.3, which="both")

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _plot_snr_timeseries(ts: lal.COMPLEX8TimeSeries, ifo: str, snr: float) -> str:
    """Create an SNR time series plot and return as base64 PNG.

    Args:
        ts: LAL complex time series containing the SNR
        ifo: Detector name for the title
        snr: Peak SNR value for annotation

    Returns:
        Base64-encoded PNG string
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    data = ts.data.data
    times = np.arange(len(data)) * ts.deltaT * 1000  # Convert to ms
    times -= times[len(times) // 2]  # Center on peak

    magnitude = np.abs(data)
    phase = np.angle(data)

    # Magnitude plot
    ax1.plot(times, magnitude, color="#E94F37", linewidth=1.5)
    ax1.axhline(
        snr, color="#666666", linestyle="--", alpha=0.5, label=f"Peak: {snr:.2f}"
    )
    ax1.set_ylabel("|SNR|", fontsize=12)
    ax1.set_title(f"{ifo} SNR Time Series", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, snr * 1.2)

    # Phase plot
    ax2.plot(times, phase, color="#44AF69", linewidth=1.5)
    ax2.set_xlabel("Time from peak (ms)", fontsize=12)
    ax2.set_ylabel("Phase (rad)", fontsize=12)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax2.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _format_gps_time(gps_sec: int, gps_ns: int) -> str:
    """Format GPS time as a string."""
    return f"{gps_sec}.{gps_ns:09d}"


def _generate_html(data: Dict, filepath: str) -> str:
    """Generate a self-contained HTML report.

    Args:
        data: Parsed coinc data from _parse_coinc_xml()
        filepath: Original file path for the title

    Returns:
        HTML string
    """
    coinc = data["coinc_inspiral"]
    triggers = data["triggers"]
    psds = data["psds"]
    snr_series = data["snr_series"]

    # Sort triggers by IFO
    triggers = sorted(triggers, key=lambda t: t.ifo)

    # Generate plots
    psd_plots = {ifo: _plot_psd(psd, ifo) for ifo, psd in psds.items()}
    snr_plots = {}
    for ifo, ts in snr_series.items():
        # Find matching trigger for peak SNR
        for trig in triggers:
            if trig.ifo == ifo:
                snr_plots[ifo] = _plot_snr_timeseries(ts, ifo, trig.snr)
                break

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coinc Event Report</title>
    <style>
        :root {{
            --primary-color: #2E86AB;
            --secondary-color: #E94F37;
            --accent-color: #44AF69;
            --bg-color: #F5F5F5;
            --card-bg: #FFFFFF;
            --text-color: #333333;
            --border-color: #DDDDDD;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), #1a5276);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            opacity: 0.9;
            font-size: 0.95em;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}

        .card h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: var(--bg-color);
            font-weight: 600;
            color: var(--primary-color);
        }}

        tr:hover {{
            background-color: #f8f9fa;
        }}

        .value {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 0.95em;
        }}

        .highlight {{
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
        }}

        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}

        .plot-container {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}

        .plot-container h3 {{
            color: var(--primary-color);
            margin-bottom: 15px;
            font-size: 1.1em;
        }}

        .plot-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}

        .ifo-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
            margin-right: 5px;
        }}

        .ifo-H1 {{ background-color: #E3F2FD; color: #1565C0; }}
        .ifo-L1 {{ background-color: #FFF3E0; color: #E65100; }}
        .ifo-V1 {{ background-color: #E8F5E9; color: #2E7D32; }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .summary-item {{
            background: var(--bg-color);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}

        .summary-item .label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}

        .summary-item .value {{
            font-size: 1.4em;
            font-weight: 600;
            color: var(--primary-color);
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .plots-grid {{
                grid-template-columns: 1fr;
            }}

            header h1 {{
                font-size: 1.5em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Gravitational Wave Event Report</h1>
            <div class="subtitle">
                <strong>File:</strong> {Path(filepath).name}<br>
                <strong>GPS Time:</strong> {
                    _format_gps_time(coinc.end_time, coinc.end_time_ns)
                }<br>
                <strong>Detectors:</strong> {coinc.ifos}
            </div>
        </header>

        <div class="card">
            <h2>Event Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">Network SNR</div>
                    <div class="value">{coinc.snr:.2f}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Chirp Mass</div>
                    <div class="value">{coinc.mchirp:.3f} Msun</div>
                </div>
                <div class="summary-item">
                    <div class="label">Total Mass</div>
                    <div class="value">{coinc.mass:.2f} Msun</div>
                </div>
                <div class="summary-item">
                    <div class="label">False Alarm Rate</div>
                    <div class="value">{coinc.combined_far:.2e} Hz</div>
                </div>
            </div>

            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>GPS Time</td>
                    <td class="value">{
                        _format_gps_time(coinc.end_time, coinc.end_time_ns)
                    }</td>
                </tr>
                <tr>
                    <td>Instruments</td>
                    <td class="value">{coinc.ifos}</td>
                </tr>
                <tr>
                    <td>Network SNR</td>
                    <td class="value">{coinc.snr:.4f}</td>
                </tr>
                <tr>
                    <td>Chirp Mass</td>
                    <td class="value">{coinc.mchirp:.6f} Msun</td>
                </tr>
                <tr>
                    <td>Total Mass</td>
                    <td class="value">{coinc.mass:.6f} Msun</td>
                </tr>
                <tr>
                    <td>Combined FAR</td>
                    <td class="value">{coinc.combined_far:.6e} Hz</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Single-Detector Triggers</h2>
            <table>
                <tr>
                    <th>Detector</th>
                    <th>End Time (GPS)</th>
                    <th>SNR</th>
                    <th>Phase (rad)</th>
                    <th>Mass 1 (Msun)</th>
                    <th>Mass 2 (Msun)</th>
                    <th>Chirp Mass (Msun)</th>
                </tr>
"""

    for trig in triggers:
        ifo_class = f"ifo-{trig.ifo}"
        gps_str = _format_gps_time(trig.end_time, trig.end_time_ns)
        html += f"""                <tr>
                    <td><span class="ifo-badge {ifo_class}">{trig.ifo}</span></td>
                    <td class="value">{gps_str}</td>
                    <td class="value">{trig.snr:.2f}</td>
                    <td class="value">{trig.coa_phase:.4f}</td>
                    <td class="value">{trig.mass1:.3f}</td>
                    <td class="value">{trig.mass2:.3f}</td>
                    <td class="value">{trig.mchirp:.4f}</td>
                </tr>
"""

    html += """            </table>
        </div>

        <div class="card">
            <h2>Power Spectral Densities</h2>
            <div class="plots-grid">
"""

    for ifo in sorted(psd_plots.keys()):
        img_data = psd_plots[ifo]
        html += f"""                <div class="plot-container">
                    <h3><span class="ifo-badge ifo-{ifo}">{ifo}</span> PSD</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{ifo} PSD">
                </div>
"""

    html += """            </div>
        </div>
"""

    if snr_plots:
        html += """        <div class="card">
            <h2>SNR Time Series</h2>
            <div class="plots-grid">
"""
        for ifo in sorted(snr_plots.keys()):
            img_data = snr_plots[ifo]
            html += f"""                <div class="plot-container">
                    <h3><span class="ifo-badge ifo-{ifo}">{ifo}</span> SNR</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{ifo} SNR">
                </div>
"""
        html += """            </div>
        </div>
"""

    html += """
        <footer>
            Generated by sgn-ligo-coinc-to-html | sgn-ligo
        </footer>
    </div>
</body>
</html>
"""

    return html


def parse_command_line():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from a coinc XML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sgn-ligo-coinc-to-html event.xml
  sgn-ligo-coinc-to-html event.xml report.html
  sgn-ligo-coinc-to-html event.xml -o custom_report.html
""",
    )
    parser.add_argument("input", help="Input coinc XML file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output HTML file (default: input name with .html extension)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        help="Output HTML file (alternative to positional argument)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_command_line()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".html")

    if args.verbose:
        print(f"Reading: {input_path}")

    data = _parse_coinc_xml(str(input_path))

    if args.verbose:
        print("Generating HTML report...")

    html = _generate_html(data, str(input_path))

    if args.verbose:
        print(f"Writing: {output_path}")

    output_path.write_text(html)

    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

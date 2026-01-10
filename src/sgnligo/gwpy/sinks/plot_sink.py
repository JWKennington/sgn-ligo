"""GWpyPlotSink: Streaming sink that generates plots at fixed intervals.

Uses the audio adapter framework for buffer accumulation with configurable
stride and overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from gwpy.timeseries import TimeSeries
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sgnts.base import Offset, TSSink

logger = logging.getLogger("sgn")


@dataclass
class GWpyPlotSink(TSSink):
    """Streaming sink that generates plots at fixed intervals.

    This sink uses the audio adapter pattern to accumulate data and generate
    plots at regular intervals during pipeline execution. Each stride of data
    produces one plot file.

    Args:
        ifo:
            Interferometer name (H1, L1, V1, etc.) for filename
        description:
            Plot description for filename (dashes will be replaced with underscores)
        plot_type:
            Type of plot to generate: "timeseries", "spectrogram", or "qtransform"
        output_dir:
            Directory to save plot files
        plot_stride:
            Time between consecutive plot starts in seconds (default 1.0).
            This determines how often a new plot is generated.
        overlap_before:
            Data to include before the stride in seconds (default 0.0).
            This is the "left" overlap - earlier time data.
        overlap_after:
            Data to include after the stride in seconds (default 0.0).
            This is the "right" overlap - later time data.
            Total plot duration = overlap_before + plot_stride + overlap_after.
            For example, stride=2, overlap_before=1, overlap_after=1 produces
            4-second plots every 2 seconds.
        input_sample_rate:
            Expected input sample rate in Hz (default 4096)
        fft_length:
            FFT length in seconds for spectrogram (default 0.5)
        qrange:
            Q range for Q-transform as (min, max) (default (4, 64))
        frange:
            Frequency range for Q-transform as (min, max) Hz (default (20, 500))
        figsize:
            Figure size as (width, height) in inches (default (12, 4))
        dpi:
            Resolution for saved plots (default 150)
        title_template:
            Custom title template. Use {ifo}, {gps_start}, {duration} placeholders.
            If None, uses default title.

    Example:
        >>> from sgnligo.gwpy.sinks import GWpyPlotSink
        >>> from sgn.apps import Pipeline
        >>>
        >>> pipeline = Pipeline()
        >>> # ... configure source ...
        >>> plot_sink = GWpyPlotSink(
        ...     name="plotter",
        ...     sink_pad_names=("in",),
        ...     ifo="H1",
        ...     description="STRAIN_FILTERED",
        ...     plot_type="spectrogram",
        ...     output_dir="./plots",
        ...     plot_stride=2.0,
        ...     overlap_before=1.0,
        ...     overlap_after=1.0,
        ... )
        >>> # ... connect and run pipeline ...
        >>> # Generates: H1-STRAIN_FILTERED-{GPS}-2.png for each stride

    Note:
        File naming follows LIGO convention:
        ``<IFO>-<DESCRIPTION>-<GPS_START>-<DURATION>.png``
        Dashes in description are replaced with underscores.
    """

    # Required parameters
    ifo: str = "H1"
    description: str = "STRAIN"

    # Plot configuration
    plot_type: str = "timeseries"
    output_dir: str = "."

    # Audio adapter configuration
    plot_stride: float = 1.0
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    input_sample_rate: int = 4096

    # Spectrogram parameters
    fft_length: float = 0.5

    # Q-transform parameters
    qrange: Tuple[float, float] = (4, 64)
    frange: Tuple[float, float] = (20, 500)

    # Plot styling
    figsize: Tuple[float, float] = (12, 4)
    dpi: int = 150
    title_template: Optional[str] = None

    # Internal state
    _plot_count: int = field(default=0, init=False, repr=False)
    _extra_padding: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        """Configure audio adapter and initialize sink."""
        # Validate parameters before configuring
        self._validate_params()

        # Configure adapter stride (how often to produce output)
        stride_offset = Offset.fromsec(self.plot_stride)

        # Configure overlap
        # overlap_before = data from before current stride (left overlap)
        # overlap_after = data from after current stride (right overlap)
        # Total plot duration = overlap_before + plot_stride + overlap_after

        # For spectrogram/qtransform, add extra padding to handle edge effects
        # from windowing. The extra padding is half the total output duration
        # on each side, which will be trimmed before plotting.
        if self.plot_type in ("spectrogram", "qtransform"):
            total_output_duration = (
                self.overlap_before + self.plot_stride + self.overlap_after
            )
            self._extra_padding = total_output_duration / 2
        else:
            self._extra_padding = 0.0

        # Calculate internal overlap (user overlap + extra padding for edge effects)
        internal_overlap_before = self.overlap_before + self._extra_padding
        internal_overlap_after = self.overlap_after + self._extra_padding

        before_samples = int(internal_overlap_before * self.input_sample_rate)
        after_samples = int(internal_overlap_after * self.input_sample_rate)

        # Set stride and overlap on adapter config
        self.adapter_config.stride = stride_offset
        self.adapter_config.overlap = (
            Offset.fromsamples(before_samples, self.input_sample_rate),
            Offset.fromsamples(after_samples, self.input_sample_rate),
        )

        # Call parent post init
        super().__post_init__()

        # Create output directory if needed
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._plot_count = 0

    def _validate_params(self) -> None:
        """Validate configuration parameters."""
        valid_types = ("timeseries", "spectrogram", "qtransform")
        if self.plot_type not in valid_types:
            raise ValueError(
                f"plot_type must be one of {valid_types}, got '{self.plot_type}'"
            )

        if self.plot_stride <= 0:
            raise ValueError(f"plot_stride must be positive, got {self.plot_stride}")

        if self.overlap_before < 0:
            raise ValueError(
                f"overlap_before must be non-negative, got {self.overlap_before}"
            )

        if self.overlap_after < 0:
            raise ValueError(
                f"overlap_after must be non-negative, got {self.overlap_after}"
            )

        if self.fft_length <= 0:
            raise ValueError(f"fft_length must be positive, got {self.fft_length}")

        if len(self.qrange) != 2 or self.qrange[0] >= self.qrange[1]:
            raise ValueError(
                f"qrange must be (min, max) with min < max, got {self.qrange}"
            )

        if len(self.frange) != 2 or self.frange[0] >= self.frange[1]:
            raise ValueError(
                f"frange must be (min, max) with min < max, got {self.frange}"
            )

    def _generate_filename(self, gps_start: float, duration: float) -> str:
        """Generate LIGO-convention filename.

        Format: <IFO>-<DESCRIPTION>-<GPS_START>-<DURATION>.png
        - Dashes only as delimiters
        - GPS_START as integer
        - DURATION as integer

        Args:
            gps_start: GPS start time in seconds
            duration: Duration in seconds

        Returns:
            Filename string following LIGO convention
        """
        # Replace any dashes in description with underscores
        safe_description = self.description.replace("-", "_")

        return f"{self.ifo}-{safe_description}-{int(gps_start)}-{int(duration)}.png"

    def _get_title(self, ts: TimeSeries, gps_start: float, duration: float) -> str:
        """Generate plot title.

        Args:
            ts: TimeSeries being plotted
            gps_start: GPS start time
            duration: Duration in seconds

        Returns:
            Title string for the plot
        """
        if self.title_template is not None:
            return self.title_template.format(
                ifo=self.ifo,
                gps_start=gps_start,
                duration=duration,
            )

        plot_type_names = {
            "timeseries": "Time Series",
            "spectrogram": "Spectrogram",
            "qtransform": "Q-Transform",
        }
        type_name = plot_type_names.get(self.plot_type, self.plot_type)
        return f"{self.ifo} {type_name} - GPS {int(gps_start)}"

    def _plot_timeseries(
        self,
        fig: Figure,
        ts: TimeSeries,
        title: str,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Plot time series waveform.

        Args:
            fig: Matplotlib figure
            ts: TimeSeries to plot
            title: Plot title
            xlim: Optional (start, end) GPS time limits for x-axis
        """
        ax = fig.add_subplot(111)
        ax.plot(ts.times.value, ts.value, linewidth=0.5, color="#1f77b4")
        ax.set_xlabel("GPS Time (s)")
        ax.set_ylabel("Strain")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if xlim is not None:
            ax.set_xlim(xlim)

    def _plot_spectrogram(
        self,
        fig: Figure,
        ts: TimeSeries,
        title: str,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Plot spectrogram.

        Args:
            fig: Matplotlib figure
            ts: TimeSeries to plot
            title: Plot title
            xlim: Optional (start, end) GPS time limits for x-axis
        """
        ax = fig.add_subplot(111)

        # Compute spectrogram
        spec = ts.spectrogram(self.fft_length)

        # Plot
        pcm = ax.pcolormesh(
            spec.times.value,
            spec.frequencies.value,
            spec.value.T,
            shading="auto",
            cmap="viridis",
        )
        ax.set_yscale("log")
        ax.set_ylim(10, ts.sample_rate.value / 2)
        ax.set_xlabel("GPS Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="ASD")
        if xlim is not None:
            ax.set_xlim(xlim)

    def _plot_qtransform(
        self,
        fig: Figure,
        ts: TimeSeries,
        title: str,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Plot Q-transform.

        Args:
            fig: Matplotlib figure
            ts: TimeSeries to plot
            title: Plot title
            xlim: Optional (start, end) GPS time limits for x-axis
        """
        ax = fig.add_subplot(111)

        # Compute Q-transform
        qtrans = ts.q_transform(qrange=self.qrange, frange=self.frange)

        # Plot
        pcm = ax.pcolormesh(
            qtrans.times.value,
            qtrans.frequencies.value,
            qtrans.value.T,
            shading="auto",
            cmap="viridis",
        )
        ax.set_yscale("log")
        ax.set_ylim(self.frange)
        ax.set_xlabel("GPS Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="Normalized Energy")
        if xlim is not None:
            ax.set_xlim(xlim)

    def _generate_plot(
        self,
        ts: TimeSeries,
        gps_start: float,
        xlim: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Generate and save plot based on plot_type.

        Args:
            ts: TimeSeries data to plot
            gps_start: GPS start time for filename
            xlim: Optional (start, end) GPS time limits for x-axis (used to
                crop view for spectrogram/qtransform after computing on
                padded data to avoid edge effects)
        """
        # For filename/title, use the visible duration (xlim) if provided
        if xlim is not None:
            visible_duration = xlim[1] - xlim[0]
        else:
            visible_duration = float(ts.duration.value)

        filename = self._generate_filename(gps_start, visible_duration)
        filepath = Path(self.output_dir) / filename

        # Create figure
        fig = Figure(figsize=self.figsize)
        FigureCanvas(fig)

        # Generate title
        title = self._get_title(ts, gps_start, visible_duration)

        # Generate plot based on type
        try:
            if self.plot_type == "timeseries":
                self._plot_timeseries(fig, ts, title, xlim=xlim)
            elif self.plot_type == "spectrogram":
                self._plot_spectrogram(fig, ts, title, xlim=xlim)
            elif self.plot_type == "qtransform":
                self._plot_qtransform(fig, ts, title, xlim=xlim)

            # Save
            fig.tight_layout()
            fig.savefig(filepath, dpi=self.dpi)

            self._plot_count += 1
            logger.getChild(self.name).info("Saved plot: %s", filepath)

        except Exception as e:
            logger.getChild(self.name).warning(
                "Failed to generate plot %s: %s", filename, e
            )
        finally:
            # Clean up figure to avoid memory leaks
            fig.clear()
            del fig

    def internal(self):
        """Process incoming frames and generate plots."""
        super().internal()

        for pad in self.sink_pads:
            frame = self.preparedframes.get(pad)
            if frame is None:
                continue

            # Check for EOS
            if frame.EOS:
                self.mark_eos(pad)

            # Skip gap frames
            if frame.is_gap:
                continue

            # Process each buffer
            for buf in frame.buffers:
                if buf.is_gap:
                    continue

                if buf.shape[0] == 0:
                    continue

                # Debug: log buffer properties
                t0_gps = Offset.tosec(buf.offset)
                buf_duration = buf.shape[0] / buf.sample_rate
                logger.getChild(self.name).debug(
                    "Buffer: offset=%s, samples=%d, rate=%d, duration=%.3f",
                    t0_gps,
                    buf.shape[0],
                    buf.sample_rate,
                    buf_duration,
                )

                # Convert buffer to TimeSeries
                ts = TimeSeries(
                    buf.data,
                    t0=t0_gps,
                    sample_rate=buf.sample_rate,
                    channel=f"{self.ifo}:{self.description}",
                )

                # For spectrogram/qtransform with extra padding, calculate the
                # plot region. The full data is used for the transform to avoid
                # edge effects, but we only plot the user-requested region.
                if self._extra_padding > 0:
                    plot_xlim = (
                        t0_gps + self._extra_padding,
                        t0_gps + buf_duration - self._extra_padding,
                    )
                    plot_t0 = plot_xlim[0]
                else:
                    plot_xlim = None
                    plot_t0 = t0_gps

                # Generate plot
                self._generate_plot(ts, plot_t0, xlim=plot_xlim)

    @property
    def plots_generated(self) -> int:
        """Return the total number of plots generated."""
        return self._plot_count

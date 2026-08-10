"""Тесты разметки и интерактивности графика распределения по глубине."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import interactive_plot  # noqa: E402


@unittest.skipUnless(interactive_plot.PLOTLY_AVAILABLE, "plotly не установлен")
class InteractivePlotTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        temp_path = Path(self.temp_dir.name)
        self.track_path = temp_path / "tracks.csv"
        self.ctd_path = temp_path / "ctd.csv"
        self.output_path = temp_path / "depth_plot"

        pd.DataFrame({
            "track_id": [1, 2],
            "object_depth_m": [13.0, 47.0],
            "class_name": ["Aurelia aurita", "Aurelia aurita"],
            "real_size_cm": [4.0, 6.0],
        }).to_csv(self.track_path, index=False)
        pd.DataFrame({
            "Depth": [0.0, 10.0, 20.0, 53.0],
            "Unused 1": [0, 0, 0, 0],
            "Unused 2": [0, 0, 0, 0],
            "Unused 3": [0, 0, 0, 0],
            "Unused 4": [0, 0, 0, 0],
            "Unused 5": [0, 0, 0, 0],
            "Temperature": [20.0, 19.0, 10.0, 9.0],
        }).to_csv(self.ctd_path, index=False)

    def _create_plot(self):
        with patch.object(
            interactive_plot.go.Figure, "write_html", autospec=True
        ) as write_html:
            interactive_plot.create_interactive_depth_plot(
                track_sizes_path=str(self.track_path),
                output_path=str(self.output_path),
                ctd_path=str(self.ctd_path),
                ctd_columns=[6],
                thermocline_mode="maximum",
            )
        figure = write_html.call_args.args[0]
        options = write_html.call_args.kwargs
        return figure, options

    def test_depth_axis_starts_at_zero_with_major_and_minor_ticks(self):
        figure, _ = self._create_plot()

        self.assertEqual(tuple(figure.layout.yaxis.range), (60.0, 0.0))
        self.assertEqual(figure.layout.yaxis.tick0, 0)
        self.assertEqual(figure.layout.yaxis.dtick, 20)
        self.assertEqual(figure.layout.yaxis.minor.tick0, 0)
        self.assertEqual(figure.layout.yaxis.minor.dtick, 10)

    def test_ctd_axis_visibility_is_linked_to_legend_trace(self):
        figure, options = self._create_plot()
        ctd_trace = next(trace for trace in figure.data if trace.meta["role"] == "ctd")
        script = options["post_script"]

        self.assertEqual(ctd_trace.meta["axis_layout_key"], "xaxis")
        self.assertIn('"axisKey": "xaxis"', script)
        self.assertIn("annotations[", script)
        self.assertIn("syncCtdAxes(traceIndices)", script)

    def test_thermocline_has_independent_legend_control(self):
        figure, options = self._create_plot()
        thermoclines = [
            trace for trace in figure.data
            if trace.meta and trace.meta.get("role") == "thermocline"
        ]

        self.assertGreater(len(thermoclines), 1)
        self.assertEqual(sum(bool(trace.showlegend) for trace in thermoclines), 1)
        self.assertIn("syncThermoclineVisibility(traceIndices)", options["post_script"])
        self.assertIn("traceIndices = eventData[1]", options["post_script"])
        self.assertNotIn("temperatureTraceIndex", options["post_script"])

    def test_html_has_svg_export_and_pdf_print_button(self):
        _, options = self._create_plot()

        self.assertEqual(options["config"]["toImageButtonOptions"]["format"], "svg")
        self.assertIn("pdfButton.textContent = 'PDF'", options["post_script"])
        self.assertIn("window.print()", options["post_script"])


if __name__ == "__main__":
    unittest.main()

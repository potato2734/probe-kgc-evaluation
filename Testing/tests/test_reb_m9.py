import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PROBE import PROBE
from vistool import VisTool


class RebM9Tests(unittest.TestCase):
    def test_all_beta_variants_share_templates_and_reduce_concentration(self):
        probabilities = np.geomspace(1, 1e6, 200)
        probabilities /= probabilities.sum()
        relation_probabilities = np.linspace(0.05, 0.95, 200)
        records = []
        for i in range(100):
            query = [i, i % 20, i + 100]
            records.append([query, "t", 1, relation_probabilities[i], probabilities[i], 0])
            records.append([query, "h", 1, relation_probabilities[i + 100], probabilities[i + 100], 0])
        met = PROBE("Fixture", "Fixture", records, 200, 20, 1, 0, 0, 1, {},
                    epsilon_entity=probabilities.min(),
                    epsilon_relation=relation_probabilities.min())
        met.calculate_final_metric = Mock(side_effect=AssertionError("m9 only needs weights"))
        vis = VisTool.__new__(VisTool)
        vis.data = "Fixture"
        saved = {}

        def capture(figure, filename, **kwargs):
            ax = figure.axes[0]
            figure.canvas.draw()
            saved[Path(filename).name] = {
                "size": figure.get_size_inches().copy(),
                "axes_bounds": ax.get_position().bounds,
                "content_bounds": ax.get_tightbbox(figure.canvas.get_renderer()).extents,
                "canvas_bounds": figure.bbox.extents,
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "scales": (ax.get_xscale(), ax.get_yscale()),
                "save_options": {key: value for key, value in kwargs.items()
                                 if key != "bbox_extra_artists"},
                "crop_bounds": kwargs["bbox_extra_artists"][0].get_bbox().bounds,
                "line_labels": [line.get_label() for line in ax.lines],
                "legend_labels": ([text.get_text() for text in ax.get_legend().get_texts()]
                                  if ax.get_legend() else []),
                "grid": [(line.get_visible(), line.get_color(), line.get_alpha())
                         for line in ax.get_ygridlines()],
                "yticks": len(ax.get_yticks()),
                "axisbelow": ax.get_axisbelow(),
                "line_style": [(line.get_color(), line.get_linestyle(), line.get_marker(),
                                line.get_linewidth()) for line in ax.lines],
                "line_x": [np.asarray(line.get_xdata()).copy() for line in ax.lines],
                "line_y": [np.asarray(line.get_ydata()).copy() for line in ax.lines],
                "histogram_x": [patch.get_xy()[:, 0].copy() for patch in ax.patches],
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", capture), \
                    patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("m9 must not write CSVs")) as csv_write, \
                    contextlib.redirect_stdout(io.StringIO()):
                concentration, mitigation = vis.reb_m9([[met]], output_dir=directory)
            csv_write.assert_not_called()
            self.assertEqual(list(Path(directory).glob("*.csv")), [])
            met.calculate_final_metric.assert_not_called()
            self.assertEqual(len(saved), 23)  # 4 x 5 diagnostics plus the 3 existing mitigation plots.
            base = concentration
            self.assertEqual(len(mitigation), 4)
            kinds = ("unit_weight", "clus_weight", "lorenz_weight", "ess_weight", "top_mass_weight")
            for suffix in ("_clip_99", "_shrink_0.1", "_log"):
                ess = saved[f"Fixture_ess_weight{suffix}.pdf"]["line_y"][0]
                top_mass = saved[f"Fixture_top_mass_weight{suffix}.pdf"]["line_y"][0]
                lorenz = saved[f"Fixture_lorenz_weight{suffix}.pdf"]
                gini = [1 - np.sum(np.diff(x) * (y[:-1] + y[1:]))
                        for x, y in zip(lorenz["line_x"], lorenz["line_y"])]
                self.assertTrue(np.all(gini <= base["gini"] + 1e-12))
                self.assertTrue(np.all(ess >= base["ess_ratio"] - 1e-12))
                self.assertTrue(np.all(top_mass <= base["top_1pct_mass"] + 1e-12))
                self.assertGreater(ess[-1], base["ess_ratio"].iloc[-1])
                for kind in kinds:
                    original = saved[f"Fixture_{kind}.pdf"]
                    variant = saved[f"Fixture_{kind}{suffix}.pdf"]
                    np.testing.assert_array_equal(original["size"], variant["size"])
                    for key in ("xlabel", "ylabel", "scales", "save_options", "line_style"):
                        self.assertEqual(original[key], variant[key])
                    for x_original, x_variant in zip(original["line_x"], variant["line_x"]):
                        np.testing.assert_array_equal(x_original, x_variant)
                    for x_original, x_variant in zip(original["histogram_x"], variant["histogram_x"]):
                        np.testing.assert_array_equal(x_original, x_variant)
                unit = saved[f"Fixture_unit_weight{suffix}.pdf"]
                self.assertEqual(len(unit["line_y"]), 5)
                np.testing.assert_allclose(unit["line_y"][0], 1 / 200)
                for weights in unit["line_y"]:
                    np.testing.assert_allclose(weights.sum(), 1)
                    self.assertTrue(np.all(weights >= 0))
            for name, figure in saved.items():
                reference = saved["Fixture_unit_weight.pdf"]
                np.testing.assert_array_equal(figure["size"], reference["size"], err_msg=name)
                np.testing.assert_allclose(figure["axes_bounds"], reference["axes_bounds"], err_msg=name)
                self.assertEqual(figure["save_options"]["bbox_inches"], "tight", name)
                np.testing.assert_allclose(figure["crop_bounds"], reference["crop_bounds"], err_msg=name)
                left, bottom, right, top = figure["content_bounds"]
                x0, y0, x1, y1 = figure["canvas_bounds"]
                self.assertGreaterEqual(min(left - x0, bottom - y0, x1 - right, y1 - top), 2, name)
                self.assertEqual(len(figure["grid"]), figure["yticks"], name)
                self.assertTrue(figure["axisbelow"], name)
                self.assertTrue(all(visible and color == "grey" and alpha == 0.7
                                    for visible, color, alpha in figure["grid"]), name)
                if "lorenz" in name:
                    self.assertNotIn("Uniform", figure["line_labels"], name)
            for suffix in ("", "_clip_99", "_shrink_0.1", "_log"):
                ess_colors = [line[0] for line in saved[f"Fixture_ess_weight{suffix}.pdf"]["line_style"]]
                top_colors = [line[0] for line in saved[f"Fixture_top_mass_weight{suffix}.pdf"]["line_style"]]
                self.assertEqual(ess_colors, ["#8c564b"])
                self.assertEqual(top_colors, ["#888888", "#555555", "#222222"])
                self.assertEqual(saved[f"Fixture_top_mass_weight{suffix}.pdf"]["legend_labels"],
                                 ["Top 10%", "Top 5%", "Top 1%"])
                self.assertEqual(len(saved[f"Fixture_lorenz_weight{suffix}.pdf"]["line_y"]), 5)
            base_unit = saved["Fixture_unit_weight.pdf"]["line_y"]
            shrunk_unit = saved["Fixture_unit_weight_shrink_0.1.pdf"]["line_y"]
            for original, shrunk in zip(base_unit, shrunk_unit):
                np.testing.assert_allclose(shrunk, 0.9 * original + 0.1 / 200)


if __name__ == "__main__":
    unittest.main()

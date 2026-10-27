import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from main import parse_args
from vistool import VisTool


class RebM16Tests(unittest.TestCase):
    def setUp(self):
        self.vis = VisTool.__new__(VisTool)
        self.vis.data = "Fixture"
        self.vis.mods = {"RotatE": "#073b4c", "ComplEx": "#118ab2"}

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def make_run(model, offset, slope):
        met = Mock(model=model)
        met.calculate_final_metric.side_effect = lambda a, b: (offset + slope * b) * (1 + a)
        return met

    def test_variance_and_std_use_alpha_one_runs_and_model_colors(self):
        for dispersion, ylabel in (("variance", "Variance / mean (%)"),
                                   ("std", "Std / mean (%)")):
            with self.subTest(dispersion=dispersion), tempfile.TemporaryDirectory() as directory:
                # Means cross across betas, so the bar order cannot identify model lines.
                mets = [[self.make_run("RotatE", 0.2, 0.0), self.make_run("RotatE", 0.4, 1.0)],
                        [self.make_run("ComplEx", 0.5, 0.0), self.make_run("ComplEx", 0.9, -1.0)]]
                saved = {}

                def capture(fig, filename, **kwargs):
                    saved[Path(filename).name] = fig.axes[0]

                with patch.object(Figure, "savefig", capture):
                    paths = self.vis.reb_m16(mets, alphas=(1.0, 0.0), betas=(0.4, 0.0, 0.2),
                                             output_dir=directory, dispersion=dispersion)
                self.assertEqual(len(paths), 7)
                self.assertEqual(len(saved), 7)
                name = f"Fixture_alpha_1.0_beta_{dispersion}.pdf"
                self.assertEqual(Path(paths[-1]).name, name)
                ax = saved[name]
                self.assertEqual(ax.get_xlabel(), r"$\beta$")
                self.assertEqual(ax.get_ylabel(), ylabel)
                self.assertEqual(ax.get_title(), self.vis.data)
                np.testing.assert_allclose(ax.figure.get_size_inches(), [4.0, 3.0])
                np.testing.assert_allclose(ax.get_position().bounds, [0.25, 0.22, 0.70, 0.70])
                self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
                self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                                 ["RotatE", "ComplEx"])
                expected_variances = ([0.08, 0.32, 0.72], [0.32, 0.08, 0.0])
                expected_means = ([0.6, 0.8, 1.0], [1.4, 1.2, 1.0])
                for line, model, variances, means in zip(
                        ax.lines, self.vis.mods, expected_variances, expected_means):
                    self.assertEqual(line.get_color(), self.vis.mods[model])
                    self.assertEqual(line.get_label(), model)
                    np.testing.assert_allclose(line.get_xdata(), [0.0, 0.2, 0.4])
                    expected = variances if dispersion == "variance" else np.sqrt(variances)
                    np.testing.assert_allclose(line.get_ydata(), np.asarray(expected) / means, atol=1e-15)
                self.assertEqual(len(ax.lines), len(mets))
                self.assertEqual(ax.yaxis.get_major_formatter()(1.0),
                                 '100.0%' if dispersion == 'variance' else '100%')
                self.assertEqual(ax.child_axes, [])
                if dispersion == 'variance':
                    self.assertEqual(ax.get_ylim(), (0.0, 0.005))
                for runs in mets:
                    for met in runs:
                        # Alpha one scores are reused from the bars.
                        self.assertEqual(met.calculate_final_metric.call_count, 6)
                        self.assertTrue(all(len(call.args) == 2
                                            for call in met.calculate_final_metric.call_args_list))

    def test_alpha_one_figure_with_single_finite_run_and_no_one_in_alphas(self):
        mets = [[self.make_run("RotatE", 0.3, 0.0), self.make_run("RotatE", np.nan, 0.0)]]
        with tempfile.TemporaryDirectory() as directory:
            paths = self.vis.reb_m16(mets, alphas=(0.5,), betas=(0.0, 0.2),
                                     output_dir=directory, file_format="svg")
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(Path(path).is_file() for path in paths))
            self.assertIn("Variance / mean (%)", Path(paths[-1]).read_text())
            for runs in mets:
                for met in runs:
                    self.assertEqual([call.args[0] for call in met.calculate_final_metric.call_args_list],
                                     [0.5, 0.5, 1.0, 1.0])

    def test_invalid_dispersion_and_all_nonfinite_scores_fail(self):
        mets = [[self.make_run("RotatE", 0.3, 0.0)]]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "dispersion"):
                self.vis.reb_m16(mets, output_dir=directory, dispersion="invalid")
            mets[0][0].calculate_final_metric.return_value = np.nan
            mets[0][0].calculate_final_metric.side_effect = None
            with self.assertRaisesRegex(ValueError, "No finite PROBE scores"):
                self.vis.reb_m16(mets, alphas=(0.0,), betas=(0.0,), output_dir=directory)

    def test_three_seed_example_plots_twenty_percent(self):
        scores = [0.5 - np.sqrt(0.1), 0.5, 0.5 + np.sqrt(0.1)]
        mets = [[Mock(model="RotatE", calculate_final_metric=Mock(return_value=score))
                 for score in scores]]
        axes = []
        with tempfile.TemporaryDirectory() as directory, patch.object(
                Figure, "savefig", lambda fig, filename, **kw: axes.append(fig.axes[0])):
            self.vis.reb_m16(mets, alphas=(1.0,), betas=(0.0,), output_dir=directory)
        np.testing.assert_allclose(axes[-1].lines[0].get_ydata(), [0.2])
        axes[-1].figure.canvas.draw()
        self.assertEqual(axes[-1].yaxis.get_major_formatter()(0.2), '20.0%')

    def test_relative_dispersion_rejects_zero_mean(self):
        mets = [[self.make_run("RotatE", 0.0, 0.0)]]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "positive mean"):
            self.vis.reb_m16(mets, alphas=(1.0,), betas=(0.0,), output_dir=directory)

    def test_cli_dispersion_selection(self):
        self.assertEqual(parse_args(["--m16"]).m16_dispersion, "variance")
        self.assertEqual(parse_args(["--m16", "--m16-dispersion", "std"]).m16_dispersion, "std")


if __name__ == "__main__":
    unittest.main()

import json

import pandas as pd

from src.pipeline import run_pipeline


def test_pipeline_quick_smoke(tmp_path):
    results = run_pipeline(quick=True, make_plots=False, n_jobs=1,
                           submissions_dir=tmp_path / "sub", reports_dir=tmp_path / "rep", verbose=False)
    sub = pd.read_csv(tmp_path / "sub" / "submission_pipeline.csv")
    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert len(sub) == 418 and sub["PassengerId"].is_unique
    assert set(sub["Survived"].unique()) <= {0, 1}
    assert 0.70 < results["holdout_metrics"]["accuracy"] < 0.90
    assert results["agreement_with_gender_baseline"] > 0.80
    assert json.loads((tmp_path / "rep" / "metrics.json").read_text())["champion"]

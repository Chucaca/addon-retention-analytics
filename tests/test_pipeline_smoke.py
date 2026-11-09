from addon_analytics.cli import run_all


def test_pipeline_smoke(tmp_path):
    f = run_all(seed=1, n_merchants=20_000, n_adopters=300, out=tmp_path)
    assert (tmp_path / "reports" / "findings.md").exists()
    for name in ["segment_profile", "lift_bars", "retention_heatmap", "churn_reasons", "rule_coverage"]:
        assert (tmp_path / "docs" / "img" / f"{name}.png").exists()
    assert 0 < f["share_direct"] <= f["share_broad"] <= 1
    assert f["ml"]["auc_gbm"] > 0.6

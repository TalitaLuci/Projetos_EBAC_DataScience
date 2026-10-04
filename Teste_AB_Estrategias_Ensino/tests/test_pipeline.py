import json

from src.pipeline import run_pipeline


def test_pipeline_end_to_end(tmp_path):
    results = run_pipeline(reports_dir=tmp_path, data_path=tmp_path / "notas.csv", make_plots=True, verbose=False)
    assert (tmp_path / "notas.csv").exists()                                 # dados são regenerados se faltarem
    assert (tmp_path / "figures" / "distribuicao_z.png").exists()
    assert (tmp_path / "figures" / "curva_poder.png").exists()
    z = results["teste_z"]
    assert z["alternative"] == "greater" and z["reject_h0"] is False
    assert json.loads((tmp_path / "results.json").read_text())["n_por_grupo_para_80pct_poder"] == 61

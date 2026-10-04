import numpy as np

from src.data import generate_samples, load_samples, to_frame


def test_samples_match_exercise_output():
    """Os 5 primeiros valores impressos no enunciado original."""
    a, b = generate_samples()
    assert np.allclose(a[:5], [87.64052346, 74.00157208, 79.78737984, 92.40893199, 88.6755799])
    assert np.allclose(b[:5], [64.25440127, 79.64282997, 68.87033835, 60.83241379, 74.66181326])
    assert len(a) == len(b) == 50


def test_csv_matches_generated_samples():
    a, b = generate_samples()
    a_csv, b_csv = load_samples()
    assert np.allclose(a, a_csv) and np.allclose(b, b_csv)


def test_long_format():
    a, b = generate_samples()
    df = to_frame(a, b)
    assert df.shape == (100, 3) and df["estrategia"].value_counts().to_dict() == {"A": 50, "B": 50}

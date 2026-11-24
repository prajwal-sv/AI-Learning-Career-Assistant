"""Test for the saved model."""

from aislca.ml import predict


def test_saved_model_makes_a_sensible_prediction():
    bundle = predict.load_model_bundle()

    result = predict.predict_one(bundle, {"G1": 10, "G2": 10, "studytime": 2})

    assert 0 <= result["estimated_mark"] <= 20
    assert 0 <= result["pass_chance"] <= 1
import os
import pytest
import tempfile
import numpy as np
from piven.utils import save_piven_model, load_piven_model


@pytest.fixture(scope="function")
def piven_model(mock_data, transformed_piven_regressor):
    x_train, _, y_train, _ = mock_data
    transformed_piven_regressor.fit(
        x_train,
        y_train,
        model__epochs=1,
        model__validation_split=0.1,
        model__batch_size=64,
    )
    return transformed_piven_regressor


class TestModelIO:
    def test_save_and_load_piven_transformedregressor(
        self, mock_data, piven_model, keras_model_function
    ):
        _, x_valid, _, y_valid = mock_data
        # Predict
        y_pred = piven_model.predict(x_valid, return_prediction_intervals=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_piven_model(piven_model, tmpdir)
            assert sorted(os.listdir(tmpdir)) == [
                "config.json",
                "piven_model.h5",
                "piven_model_config.json",
                "piven_ttr.joblib",
            ]
            piven_model_loaded = load_piven_model(keras_model_function, tmpdir)
        y_pred_loaded = piven_model_loaded.predict(
            x_valid, return_prediction_intervals=False
        )
        assert np.all(y_pred == y_pred_loaded)

    def test_save_and_load_piven_pipeline(
        self, mock_data, piven_model, keras_model_function
    ):
        _, x_valid, _, y_valid = mock_data
        # Predict
        y_pred = piven_model.regressor_.predict(
            x_valid, return_prediction_intervals=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_piven_model(piven_model.regressor_, tmpdir)
            assert sorted(os.listdir(tmpdir)) == [
                "config.json",
                "piven_model.h5",
                "piven_model_config.json",
                "piven_pipeline.joblib",
            ]
            piven_model_loaded = load_piven_model(keras_model_function, tmpdir)
        y_pred_loaded = piven_model_loaded.predict(
            x_valid, return_prediction_intervals=False
        )
        assert np.all(y_pred == y_pred_loaded)

    def test_save_and_load_piven_wrapper(
        self, mock_data, piven_model, keras_model_function
    ):
        _, x_valid, _, y_valid = mock_data
        # Predict
        y_pred = piven_model.regressor_["model"].predict(
            x_valid, return_prediction_intervals=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            save_piven_model(piven_model.regressor_["model"], tmpdir)
            assert sorted(os.listdir(tmpdir)) == [
                "config.json",
                "piven_model.h5",
                "piven_model_config.json",
            ]
            piven_model_loaded = load_piven_model(keras_model_function, tmpdir)
        y_pred_loaded = piven_model_loaded.predict(
            x_valid, return_prediction_intervals=False
        )
        assert np.all(y_pred == y_pred_loaded)

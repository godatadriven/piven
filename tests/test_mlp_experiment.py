import tempfile
from pathlib import Path
import pytest
import tensorflow as tf
import numpy as np
from piven.experiments import PivenMlpExperiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


@pytest.fixture(scope="function")
def experiment():
    return PivenMlpExperiment(
        input_dim=1,
        dense_units=(64, 64),
        dropout_rate=(0.1, 0.1),
        lambda_=25.0,
        bias_init_low=-3.0,
        bias_init_high=3.0,
        lr=0.0001,
    )


@pytest.fixture(scope="function")
def mock_data_categorical_column():
    # Make mock data with categorical input variable
    n_samples = 500
    x1 = np.random.uniform(low=-2.0, high=2.0, size=(n_samples, 1))
    x2 = np.random.choice([0, 1, 2, 3, 4], n_samples)
    y = (
        1.5 * np.sin(np.pi * x1[:, 0])
        + 0.8 * x2
        + np.random.normal(loc=0.0, scale=1 * np.power(x1[:, 0], 2))
    )
    x = np.column_stack((x1, x2))
    return x, y


@pytest.fixture(scope="function")
def sklearn_preprocessing_pipeline():
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("scaler", StandardScaler(), [0]), ("ohe", OneHotEncoder(), [1])]
                ),
            )
        ]
    )


class TestPivenMlpExperiment:
    def test_experiment_build_model(self, experiment):
        experiment.build_model()

    def test_experiment_fit_model_no_preprocess(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(
            x_train,
            y_train,
            model__epochs=3,
            model__validation_split=0.1,
            model__callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
        )

    def test_experiment_fit_model_with_preprocess(
        self, mock_data_categorical_column, sklearn_preprocessing_pipeline
    ):
        x, y = mock_data_categorical_column
        # Start an experiment
        # The only issue with the preprocessing pipeline is that the
        # number of features that was specified when building the model
        # no longer works, so we need to think about this and compute it manually.
        experiment = PivenMlpExperiment(
            input_dim=6,  # 1 + 5 columns after applying ohe
            dense_units=(64, 64),
            dropout_rate=(0.1, 0.1),
            lambda_=25.0,
            bias_init_low=-3.0,
            bias_init_high=3.0,
            lr=0.0001,
        )
        experiment.build_model(preprocess=sklearn_preprocessing_pipeline)
        experiment.fit(
            x,
            y,
            model__epochs=3,
            model__validation_split=0.1,
            model__callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
        )

    def test_experiment_io(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(
            x_train,
            y_train,
            model__epochs=3,
            model__callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=1)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment.save(tmpdir)
            assert (Path(tmpdir) / "experiment_params.json").is_file()
            _ = PivenMlpExperiment.load(path=tmpdir)

    def test_experiment_scoring(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(x_train, y_train, model__epochs=3)
        y_pred, y_pi_low, y_pi_high = experiment.predict(
            x_valid, return_prediction_intervals=True
        )
        scores = experiment.score(y_valid, y_pred, y_pi_low, y_pi_high)
        assert sorted(list(scores.keys())) == [
            "coverage",
            "loss",
            "mae",
            "pi_width",
            "rmse",
        ]

    def test_experiment_logging(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(x_train, y_train, model__epochs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment.log(x_valid, y_valid, tmpdir, model=True, predictions=True)
            f = sorted([str(cf).split("/")[-1] for cf in Path(tmpdir).glob("*")])
            assert f == ["metrics.json", "model", "predictions.csv"]

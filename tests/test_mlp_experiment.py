import tempfile
from pathlib import Path
import pytest
import tensorflow as tf
from piven.experiments import PivenMlpExperiment


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


class TestPivenMlpExperiment:
    def test_experiment_build_model(self, experiment):
        experiment.build_model()

    def test_experiment_fit_model(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(
            x_train,
            y_train,
            model__epochs=3,
            model__validation_split=0.1,
            model__callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
        )

    def test_experiment_io(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(x_train, y_train, model__epochs=3)
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
        assert sorted(list(scores.keys())) == ["coverage", "loss", "pi_width"]

    def test_experiment_logging(self, mock_data, experiment):
        x_train, x_valid, y_train, y_valid = mock_data
        experiment.build_model()
        experiment.fit(x_train, y_train, model__epochs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment.log(x_valid, y_valid, tmpdir, model=True, predictions=True)
            f = sorted([str(cf).split("/")[-1] for cf in Path(tmpdir).glob("*")])
            assert f == ["metrics.json", "model", "predictions.csv"]

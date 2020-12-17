import pytest
import tempfile
import json
from pathlib import Path
import tensorflow as tf
from piven.regressors import PivenRegressor
from piven.metrics import picp, mpiw, coverage, pi_distance
import joblib
from piven.transformers import PivenTransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


@pytest.fixture(scope="class")
def optimizer(request):
    request.cls.optimizer = tf.keras.optimizers.Adam(lr=0.0007)


@pytest.fixture(scope="function")
def piven_model() -> PivenRegressor:
    return PivenRegressor(
        input_dim=1,
        optimizer=tf.keras.optimizers.Adam(lr=0.0007),
        dense_units=(128, 128),
        dropout_rate=(0.05, 0.05),
        activation="relu",
        epochs=600,
        batch_size=128,
        validation_split=0.1,
        verbose=True,
        callbacks=None,
        metrics=[picp, mpiw],
        reg=25.0,
        soften=160.0,
        alpha=0.05,
    )


@pytest.fixture(scope="function")
def pipeline(piven_model) -> PivenTransformedTargetRegressor:
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", piven_model)])
    return PivenTransformedTargetRegressor(
        regressor=pipeline, transformer=StandardScaler()
    )


@pytest.mark.usefixtures("optimizer")
class TestPivenRegressor:
    def test_fail_with_unequal_dense_and_dropout(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(64, 64),
                dropout_rate=(0.1,),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
            )

    def test_fail_dropout_not_float(self):
        with pytest.raises(TypeError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(64, 64),
                dropout_rate=(0.1, 0),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
            )

    def test_fail_dropout_not_all_in_zero_one_range(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(64, 64),
                dropout_rate=(-0.2, 0.9),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
            )

    def test_fail_dense_units_not_integers(self):
        with pytest.raises(TypeError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(64.2, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
            )

    def test_all_dense_units_larger_than_zero(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
            )

    def test_callbacks_list(self):
        early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=self.optimizer,
                dense_units=(64, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=early_stop,
            )

    def test_model_alpha_not_in_range(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=15.0,
                soften=160.0,
                alpha=1.2,
            )

    def test_model_alpha_not_float(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=15.0,
                soften=160.0,
                alpha=0,
            )

    def test_model_reg_not_in_range(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=-1,
                soften=160.0,
                alpha=0.05,
            )

    def test_model_reg_not_float(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=15,
                soften=160.0,
                alpha=0.05,
            )

    def test_model_soften_not_in_range(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=15.0,
                soften=-1,
                alpha=0.05,
            )

    def test_model_soften_not_float(self):
        with pytest.raises(ValueError):
            _ = PivenRegressor(
                input_dim=2,
                optimizer=optimizer,
                dense_units=(-1, 64),
                dropout_rate=(0.1, 0.1),
                activation="relu",
                epochs=100,
                batch_size=128,
                validation_split=0.1,
                verbose=False,
                callbacks=None,
                reg=15.0,
                soften=160,
                alpha=0.05,
            )

    def test_create_successful_model(self):
        _ = PivenRegressor(
            input_dim=2,
            optimizer=self.optimizer,
            dense_units=(64, 64),
            dropout_rate=(0.1, 0.1),
            activation="relu",
            epochs=100,
            batch_size=128,
            validation_split=0.1,
            verbose=False,
            callbacks=None,
        )

    def test_fit_piven(self, piven_model, mock_data):
        x_train, x_valid, y_train, y_valid = mock_data
        piven_model.fit(x_train, y_train)
        yhat, pi_low, pi_high = piven_model.predict(
            x_valid, return_prediction_intervals=True
        )
        xlinspace = np.linspace(-2, 2, x_train.shape[0])
        pred, lower, upper = piven_model.predict(
            xlinspace.reshape(-1, 1), return_prediction_intervals=True
        )
        # Plot if wanted in debug console
        # import matplotlib.pyplot as plt
        # plt.scatter(X_train, y_train[:, 0]);
        # plt.scatter(X_valid, y_valid[:, 0]);
        # plt.fill_between(xlinspace, lower, upper, color="blue", alpha=0.3);
        # plt.show()
        rmse = np.sqrt(mean_squared_error(y_valid[:, 0], yhat))
        assert rmse < 2
        cov = coverage(y_valid[:, 0], pi_low, pi_high)
        assert 0.85 < cov < 0.98
        # Distance between PIs
        pidist_lower = pi_distance(lower[xlinspace < -0.5], upper[xlinspace < -0.5])
        pidist_middle = pi_distance(
            lower[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
            upper[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
        )
        pidist_upper = pi_distance(lower[xlinspace > 0.5], upper[xlinspace > 0.5])
        assert pidist_upper > pidist_middle
        assert pidist_lower > pidist_middle
        # Test can save
        with tempfile.TemporaryDirectory() as tempdir:
            piven_model.save(tempdir)
            mod = PivenRegressor.load(tempdir)
        # Predict
        yhat_from_saved, pi_low_from_saved, pi_high_from_saved = mod.predict(
            x_valid, return_prediction_intervals=True
        )
        assert np.all(yhat_from_saved == yhat)
        assert np.all(pi_low_from_saved == pi_low)
        assert np.all(pi_high_from_saved == pi_high)

    def test_fit_piven_in_sklearn_pipeline(self, pipeline, mock_data):
        x_train, x_valid, y_train, y_valid = mock_data
        pipeline.fit(x_train, y_train[:, 0].reshape(-1, 1))
        yhat, pi_low, pi_high = pipeline.predict(
            x_valid, return_prediction_intervals=True
        )
        xlinspace = np.linspace(-2, 2, x_train.shape[0])
        pred, lower, upper = pipeline.predict(
            xlinspace.reshape(-1, 1), return_prediction_intervals=True
        )
        # Plot if wanted in debug console
        # import matplotlib.pyplot as plt
        # plt.scatter(x_train, y_train[:, 0]);
        # plt.scatter(X_valid, y_valid[:, 0]);
        # plt.fill_between(xlinspace, lower, upper, color="blue", alpha=0.3);
        # plt.show()
        rmse = np.sqrt(mean_squared_error(y_valid[:, 0], yhat))
        assert rmse < 2
        cov = coverage(y_valid[:, 0], pi_low, pi_high)
        assert 0.85 < cov < 0.99
        # Distance between PIs
        pidist_lower = pi_distance(lower[xlinspace < -0.5], upper[xlinspace < -0.5])
        pidist_middle = pi_distance(
            lower[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
            upper[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
        )
        pidist_upper = pi_distance(lower[xlinspace > 0.5], upper[xlinspace > 0.5])
        assert pidist_upper > pidist_middle
        assert pidist_lower > pidist_middle

    def test_save_piven_transformed_target_regressor(self, piven_model, mock_data):
        """Save and load a transformed target regressor"""
        # NB: not making this a method for PivenTransformedTargetRegressor because
        #      the procedure is dependent on what is passed as a regressor.
        x_train, _, y_train, _ = mock_data
        piven_model.epochs = 10
        # Set up a standard scaler outside of a model pipeline used by piven.
        #  this is done to mock the situation in 'experiments/train_piven_bert.py'
        scaler = StandardScaler()
        x_train_transformed = scaler.fit_transform(x_train)
        pipeline = PivenTransformedTargetRegressor(
            regressor=piven_model, transformer=StandardScaler()
        )
        pipeline.fit(x_train_transformed, y_train[:, 0])
        # Predict
        y_pred = pipeline.predict(x_train_transformed)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pipeline.regressor_.save(tmpdir)
            # Unset the regressor
            pipeline.regressor_ = None
            pipeline.regressor = None
            joblib.dump(pipeline, tmpdir / "TTR.joblib")
            joblib.dump(scaler, tmpdir / "scaler.joblib")
            # Now load them back in and create a new pipeline
            model = PivenRegressor.load(tmpdir)
            # Load model config
            with (tmpdir / "config.json").open("r") as infile:
                piven_config = json.load(infile)
            # Load the pipeline from disk
            pipeline_from_disk = joblib.load(tmpdir / "TTR.joblib")
            scaler_from_disk = joblib.load(tmpdir / "scaler.joblib")
        piven_config["optimizer"] = model.optimizer
        piven_config["metrics"] = model.metrics
        piven_config["callbacks"] = model.callbacks
        pipeline_from_disk.regressor = PivenRegressor(**piven_config)
        pipeline_from_disk.regressor_ = make_pipeline(scaler_from_disk, model)
        # Predict
        y_pred_from_disk = pipeline_from_disk.predict(x_train)
        # Assert equal
        assert np.all(y_pred == y_pred_from_disk)

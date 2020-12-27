from pathlib import Path
import logging
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from piven.experiments import PivenMlpExperiment

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def run_piven(
    input_file: str,
    target: str,
    output_dir: str,
    dense_units: str,
    dropout_rate: str,
    pi_init_low: float = typer.Option(
        -3.0, help="initial value of the pi lower bound."
    ),
    pi_init_high: float = typer.Option(
        3.0, help="initial value of the pi upper bound."
    ),
    learning_rate: float = typer.Option(
        0.0001, help="learning rate to be used with the Adam optimizer."
    ),
    lambda_: float = typer.Option(
        25.0, help="lambda parameter used in the piven loss function."
    ),
    epochs: int = typer.Option(25, help="number of epochs the model will be trained."),
    validation_split: float = typer.Option(
        0.1, help="validation split to use for keras training."
    ),
    batch_size: int = typer.Option(64, help="batch size for keras training."),
    verbose: bool = typer.Option(
        True, help="if true, output results of training keras model."
    ),
):
    """
    Run a piven model from the command line
    """
    dense_units = tuple([int(du) for du in dense_units.split("-")])
    dropout_rate = tuple([float(dr) for dr in dropout_rate.split("-")])
    # Check output dir now
    poutput_dir = Path(output_dir)
    if not poutput_dir.is_dir():
        raise NotADirectoryError(f"Directory {output_dir} does not exist.")
    logger.info("Loading data ...")
    # Load input file
    input_data = pd.read_csv(input_file)
    if target not in input_data.columns:
        raise KeyError(f"Target {target} not found in dataset.")
    y = input_data[target].values
    x = input_data.loc[:, input_data.columns != target].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    logger.info("Setting up experiment ...")
    experiment = PivenMlpExperiment(
        input_dim=x.shape[-1],
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        lambda_=lambda_,
        bias_init_low=pi_init_low,
        bias_init_high=pi_init_high,
        lr=learning_rate,
    )
    experiment.build_model()
    logger.info("Start experiment ...")
    experiment.fit(
        x_train,
        y_train,
        model__epochs=epochs,
        model__validation_split=validation_split,
        model__batch_size=batch_size,
        model__verbose=verbose,
    )
    y_pred, y_pi_low, y_pi_high = experiment.predict(
        x_test, return_prediction_intervals=True
    )
    # Retrieve output metrics
    metrics = experiment.score(y_test, y_pred, y_pi_low, y_pi_high)
    metrics_string = " | ".join(
        [f"{metric}={metric_value}" for metric, metric_value in metrics.items()]
    )
    logger.info(
        f"Finished experiment. Results on test split are: {metrics_string}."
        + f" Logging model output to {output_dir}."
    )
    # Log metrics, model, predictions
    experiment.log(x_test, y_test, path=output_dir, model=True, predictions=True)
    return experiment

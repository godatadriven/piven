import tempfile
from pathlib import Path
import typer.testing
import pandas as pd
import numpy as np
from piven.cli import app


def test_cli(mock_data):
    sh = typer.testing.CliRunner()
    x_train, _, y_train, _ = mock_data
    d = pd.DataFrame(
        data=np.column_stack([x_train, y_train]), columns=["feature", "target"]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        p = Path(tempdir)
        (p / "output").mkdir()
        # Dump data to file so can call it using click app
        d.to_csv(p / "data.csv")
        r = sh.invoke(
            app,
            [
                str(p / "data.csv"),
                "target",
                str(p / "output"),
                "64-64",
                "0.1-0.1",
                "--epochs",
                10,
            ],
        )
        assert r.exit_code == 0

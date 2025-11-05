"""Experiment tracking utilities with W&B and simple logging fallback."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracker supporting W&B and local logging.

    Example:
        >>> tracker = ExperimentTracker(
        ...     experiment_name="activation_patching",
        ...     config={"model": "gpt2-medium", "task": "fact_tracing"},
        ...     use_wandb=True,
        ... )
        >>> tracker.log({"loss": 0.5, "accuracy": 0.85})
        >>> tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        config: dict[str, Any] | None = None,
        use_wandb: bool = False,
        wandb_project: str = "mech-interp",
        log_dir: str = "results/logs",
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name for this experiment (auto-generated if None)
            config: Configuration dictionary to log
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            log_dir: Directory for local logs
        """
        self.config = config or {}
        self.use_wandb = use_wandb

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        self.experiment_name = experiment_name

        # Setup local logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"

        # Initialize W&B if requested
        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=self.config,
                )
                logger.info(f"W&B initialized: {wandb_project}/{experiment_name}")
            except ImportError:
                logger.warning("wandb not installed, falling back to local logging only")
                self.use_wandb = False
        else:
            logger.info(f"Local logging to {self.log_file}")

        # Save config locally
        config_file = self.log_dir / f"{experiment_name}_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Experiment '{experiment_name}' initialized")

    def log(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
    ) -> None:
        """
        Log metrics to W&B and/or local file.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step/iteration number
        """
        # Add timestamp
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics,
        }

        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        # Log to local file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_table(
        self,
        name: str,
        data: list,
        columns: list | None = None,
    ) -> None:
        """
        Log a table of data.

        Args:
            name: Table name
            data: List of rows (each row is a list or dict)
            columns: Optional column names (required if data rows are lists)
        """
        if self.use_wandb:
            import wandb

            if columns is not None:
                table = wandb.Table(data=data, columns=columns)
            else:
                # Assume data is list of dicts
                columns = list(data[0].keys())
                table = wandb.Table(
                    data=[[row[col] for col in columns] for row in data],
                    columns=columns,
                )
            self.wandb.log({name: table})

        # Also save locally as JSON
        table_file = self.log_dir / f"{self.experiment_name}_{name}.json"
        with open(table_file, "w") as f:
            json.dump({"columns": columns, "data": data}, f, indent=2)

    def log_artifact(
        self,
        file_path: str,
        artifact_name: str | None = None,
        artifact_type: str = "model",
    ) -> None:
        """
        Log a file artifact (model checkpoint, dataset, etc.).

        Args:
            file_path: Path to file to log
            artifact_name: Name for artifact (uses filename if None)
            artifact_type: Type of artifact (model, dataset, results, etc.)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Artifact file not found: {file_path}")
            return

        if artifact_name is None:
            artifact_name = file_path.name

        if self.use_wandb:
            import wandb

            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
            )
            artifact.add_file(str(file_path))
            self.wandb.log_artifact(artifact)

        logger.info(f"Logged artifact: {artifact_name} ({artifact_type})")

    def log_figure(
        self,
        name: str,
        figure,
        step: int | None = None,
    ) -> None:
        """
        Log a matplotlib or plotly figure.

        Args:
            name: Figure name
            figure: Matplotlib or Plotly figure object
            step: Optional step number
        """
        if self.use_wandb:
            self.wandb.log({name: figure}, step=step)

        # Save locally
        figure_file = self.log_dir / f"{self.experiment_name}_{name}.html"
        try:
            # Try plotly first
            figure.write_html(str(figure_file))
        except AttributeError:
            # Fall back to matplotlib
            figure_file = self.log_dir / f"{self.experiment_name}_{name}.png"
            figure.savefig(str(figure_file))

        logger.info(f"Saved figure: {figure_file}")

    def finish(self) -> None:
        """Finish the experiment and cleanup."""
        if self.use_wandb:
            self.wandb.finish()

        logger.info(f"Experiment '{self.experiment_name}' finished")
        logger.info(f"Logs saved to: {self.log_dir}")


def load_experiment_logs(log_file: str) -> list:
    """
    Load experiment logs from a JSONL file.

    Args:
        log_file: Path to JSONL log file

    Returns:
        List of log entries (dicts)
    """
    logs = []
    with open(log_file) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


# Simple context manager interface
class track_experiment:
    """
    Context manager for experiment tracking.

    Example:
        >>> with track_experiment("my_experiment", config={"lr": 0.001}) as tracker:
        ...     for step in range(100):
        ...         loss = train_step()
        ...         tracker.log({"loss": loss}, step=step)
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        config: dict[str, Any] | None = None,
        use_wandb: bool = False,
        wandb_project: str = "mech-interp",
        log_dir: str = "results/logs",
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.log_dir = log_dir
        self.tracker = None

    def __enter__(self) -> ExperimentTracker:
        self.tracker = ExperimentTracker(
            experiment_name=self.experiment_name,
            config=self.config,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            log_dir=self.log_dir,
        )
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            self.tracker.finish()

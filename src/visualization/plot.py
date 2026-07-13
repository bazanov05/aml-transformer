from src.tracker.db import connection
import seaborn as sns
from src.tracker.db.epochs import(
    get_best_epoch_per_run,
    get_epochs_for_run
)
import pandas as pd
import matplotlib.pyplot as plt


def val_loss_per_config_run(num_of_runs: int, file_for_plot: str):
    """
    Retrieves training data for a specified number of runs and generates 
    a line plot of validation loss across epochs.

    Args:
        num_of_runs (int): The total number of runs to visualize.
        file_for_plot (str): The file path to save the generated plot.
    """
    connection.init_pool()
    data_from_trainigs = []

    with connection.pool.connection() as conn:
        for run in range(1, num_of_runs + 1):
            data_from_trainigs.extend(get_epochs_for_run(conn=conn, run_id=run))

    data_from_trainigs = pd.DataFrame(data=data_from_trainigs)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_from_trainigs, x="epoch_num", y="val_loss", hue="run_id", palette="tab20")
    plt.title("Validation Loss Tracking per Config Run")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(fname=file_for_plot)
    plt.show()

    connection.close_pool()


def best_epoch_per_each_run(file_for_plot: str):
    """
    Retrieves the best epoch (lowest validation loss) for every recorded run 
    and generates a bar plot comparing their performance.

    Args:
        file_for_plot (str): The file path to save the generated plot.
    """
    connection.init_pool()
    data_from_trainigs = []

    with connection.pool.connection() as conn:
        data_from_trainigs.extend(get_best_epoch_per_run(conn=conn))
    data_from_trainigs = pd.DataFrame(data=data_from_trainigs)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data_from_trainigs, x="run_id", y="val_loss", palette="tab20")    
    plt.title("Best epochs per each run")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(fname=file_for_plot)
    plt.show()

    connection.close_pool()

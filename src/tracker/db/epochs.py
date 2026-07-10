def insert_epoch(conn, run_id: int, epoch_num: int, train_loss: float, val_loss: float) -> int:
    """
    Inserts a new epoch record for a given run.

    Args:
        conn - connection to Postgres server inside Docker.
        run_id - id of the run this epoch belongs to.
        epoch_num - the epoch number in the training loop.
        train_loss - training loss for this epoch.
        val_loss - validation loss for this epoch.

    Returns:
        id of the newly created epoch row.
    """
    cursor = conn.cursor()

    cursor.execute(
    """
    INSERT INTO epochs
    (run_id, epoch_num, train_loss, val_loss)
    VALUES (%s, %s, %s, %s)
    RETURNING id
    """, (run_id, epoch_num, train_loss, val_loss)
    )

    conn.commit()

    return cursor.fetchone()["id"]


def get_epochs_for_run(conn, run_id: int):
    """
    Fetches all epochs for a given run ordered by epoch number.

    Args:
        conn - connection to Postgres server inside Docker.
        run_id - id of the run to fetch epochs for.

    Returns:
        List of Dicts, each Dict representing one epoch as
        {id, run_id, epoch_num, train_loss, val_loss, created_at}.
        Returns empty list if no epochs exist for that run.
    """
    cursor = conn.cursor()

    cursor.execute(
    """
    SELECT * FROM epochs
    WHERE run_id = %s
    ORDER BY epoch_num ASC
    """, (run_id,)
    )

    return cursor.fetchall()


def get_best_epoch_per_run(conn):
    """
    Fetches the best epoch for each run based on lowest validation loss.

    Args:
        conn - connection to Postgres server inside Docker.

    Returns:
        List of Dicts, each Dict representing the best epoch per run as
        {run_id, epoch_num, val_loss}. One row per run, ordered by run_id.
        Returns empty list if no epochs exist.
    """
    cursor = conn.cursor()
    
    # DISTINCT ON keeps one unique row per run_id determined by ORDER BY 
    cursor.execute(
    """
    SELECT DISTINCT ON (run_id)     
        run_id,
        epoch_num, 
        val_loss 
    FROM epochs
    ORDER BY run_id, val_loss ASC
    """
    )

    return cursor.fetchall()

def create_run(conn, d_model: int, lr: float, num_of_blocks: int, num_of_heads: int) -> int:
    """
    Inserts new run into the run table.

    Args:
        conn - connection to Postgres server inside Docker.
        d_model - how many features each token has.
        lr  - learning rate.
        num_of_blocks - number of Transformer blocks.
        num_of_heads - number of heads in Attention Block.
    
    Returns:
        id of new created run in db.
    """
    cursor = conn.cursor() # create cursor 
    
    cursor.execute(
        """
        INSERT INTO runs (d_model, lr, num_of_blocks, num_of_heads) 
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (d_model, lr, num_of_blocks, num_of_heads)     
    )

    conn.commit()   # confirm the INSERT 

    id = cursor.fetchone()[0] # fetch the returned id

    return id


def get_run(conn, id: int):
    """
    Fetches a single run from the database by id.

    Args:
        conn - connection to Postgres server inside Docker.
        id - id of the run to fetch.

    Returns:
        Tuple of (id, d_model, lr, num_of_blocks, num_of_heads, created_at)
        or None if no run with that id exists.
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM runs 
        WHERE id = %s
        """, (id,)
    )

    return cursor.fetchone()


def get_all_runs(conn):
    """
    Fetches all runs from the database.

    Args:
        conn - connection to Postgres server inside Docker.

    Returns:
        List of tuples, each tuple representing one run as
        (id, d_model, lr, num_of_blocks, num_of_heads, created_at).
        Returns empty list if no runs exist.
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM runs 
        ORDER BY created_at DESC
        """
    )

    return cursor.fetchall()


def get_best_run(conn):
    """
    Fetches the run with the lowest validation loss across all epochs.

    Args:
        conn - connection to Postgres server inside Docker.

    Returns:
        Tuple of (run_id, val_loss) for the best performing run,
        or None if no epochs exist yet.
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT run_id, val_loss FROM runs r
        INNER JOIN epochs e
        ON r.id = e.run_id
        ORDER BY e.val_loss ASC
        LIMIT 1
        """
    )

    return cursor.fetchone()

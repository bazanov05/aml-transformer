from fastapi import APIRouter, Depends, HTTPException
from src.tracker.api.schemas import EpochCreate, EpochResponse, BestEpochResponse
from src.tracker.db.connection import get_db
from src.tracker.db.epochs import (
    get_best_epoch_per_run,
    get_epochs_for_run,
    insert_epoch
)
from typing import List


epoch_router = APIRouter()


@epoch_router.post(path="/runs/{run_id}/epochs")
def insert_new_epoch(data: EpochCreate, run_id: int, db=Depends(get_db)) -> int:
    """
    Logs a new epoch for a specific training run.

    Args:
        data: EpochCreate schema containing epoch_num, train_loss, val_loss.
        run_id: ID of the run this epoch belongs to, taken from the URL.
        db: Database connection injected by FastAPI via Depends.

    Returns:
        id of the newly created epoch row.
    """
    return insert_epoch(db, run_id, **data.model_dump())


@epoch_router.get(path="/epochs/best", response_model=List[BestEpochResponse])
def get_best_epoch_for_every_run(db=Depends(get_db)):
    """
    Fetches the best epoch for every run based on lowest validation loss.

    Args:
        db: Database connection injected by FastAPI via Depends.

    Returns:
        List of BestEpochResponse objects, one per run, each containing
        run_id, epoch_num and val_loss of the best epoch.

    Raises:
        HTTPException: 404 error if no epochs exist in the database.
    """
    epochs = get_best_epoch_per_run(conn=db)

    if not epochs:
        raise HTTPException(status_code=404, detail="No epochs were found, which means no training runs were made")
    
    return epochs


@epoch_router.get(path="/runs/{run_id}/epochs", response_model=List[EpochResponse])
def get_all_epochs_for_single_run(run_id: int, db=Depends(get_db)):
    """
    Fetches all epochs for a specific training run ordered by epoch number.

    Args:
        run_id: ID of the run to fetch epochs for, taken from the URL.
        db: Database connection injected by FastAPI via Depends.

    Returns:
        List of EpochResponse objects ordered by epoch_num ascending.

    Raises:
        HTTPException: 404 error if no epochs exist for this run.
    """
    epochs = get_epochs_for_run(conn=db, run_id=run_id)

    if not epochs:
        raise HTTPException(status_code=404, detail="No epochs were found for this run")
    
    return epochs

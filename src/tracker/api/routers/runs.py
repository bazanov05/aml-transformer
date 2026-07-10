from fastapi import APIRouter, Depends, HTTPException
from src.tracker.api.schemas import RunCreate, RunResponse, BestRunResponse
from src.tracker.db.connection import get_db
from src.tracker.db.runs import (
    create_run,
    get_all_runs,
    get_best_run,
    get_run
)
from typing import List


run_router = APIRouter()


@run_router.post(path="/runs")
def create_new_run(data: RunCreate, db = Depends(get_db)) -> int:
    """
    Creates a new training run and saves it to the database.

    Args:
        data: RunCreate schema containing d_model, lr, num_of_blocks, num_of_heads.
        db: Database connection injected by FastAPI via Depends.

    Returns:
        id of the newly created run.
    """
    # create the dict from RunCreate obj and unpack it to method's args
    return create_run(db, **data.model_dump())


@run_router.get(path="/runs", response_model=List[RunResponse])
def get_runs(db=Depends(get_db)):
    """
    Fetches all training runs from the database.

    Args:
        db: Database connection injected by FastAPI via Depends.

    Returns:
        List of RunResponse objects ordered by creation date descending.

    Raises:
        HTTPException: 404 error if no runs exist in the database.
    """
    all_runs = get_all_runs(conn=db)

    if not all_runs:
        raise HTTPException(status_code=404, detail="No runs were found in db")
    
    return all_runs


@run_router.get(path="/runs/best", response_model=BestRunResponse)
def get_the_best_run(db=Depends(get_db)):
    """
    Fetches the training run with the lowest validation loss across all epochs.

    Args:
        db: Database connection injected by FastAPI via Depends.

    Returns:
        RunResponse object of the best performing run.

    Raises:
        HTTPException: 404 error if no runs and epochs exist yet.
    """
    best_run = get_best_run(conn=db)

    if best_run is None:
        raise HTTPException(status_code=404, detail="The best run was not found")

    return best_run


@run_router.get(path="/runs/{id}", response_model=RunResponse)
def get_the_run_by_id(id: int, db = Depends(get_db)) -> RunResponse:
    """
    Fetches a specific training run by its ID.

    Args:
        id: The unique identifier of the run.
        db: Database connection injected by FastAPI.

    Returns:
        RunResponse object containing the run details.
        (id, d_model, lr, num_of_blocks, num_of_heads, created_at)

    Raises:
        HTTPException: 404 error if the run does not exist in the database.
    """
    run = get_run(conn=db, id=id)

    if run is None:
        raise HTTPException(status_code=404, detail="Run with this id does not exist")

    return RunResponse(**run)

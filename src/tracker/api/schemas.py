from pydantic import BaseModel
from datetime import datetime


class RunCreate(BaseModel):
    d_model: int
    lr: float
    num_of_blocks: int
    num_of_heads: int


class RunResponse(BaseModel):
    id: int
    d_model: int
    lr: float
    num_of_blocks: int
    num_of_heads: int
    created_at: datetime


class EpochCreate(BaseModel):
    epoch_num: int
    train_loss: float
    val_loss: float


class EpochResponse(BaseModel):
    id: int
    run_id: int
    epoch_num: int
    train_loss: float
    val_loss: float
    created_at: datetime

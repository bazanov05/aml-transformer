from src.tracker.api.routers.epochs import epoch_router
from src.tracker.api.routers.runs import run_router
from fastapi import FastAPI
from src.tracker.db.connection import init_pool, close_pool
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()     # creates connection pool and connects to PostgreSQL via Docker
    yield   # gives pool to app and freezes here
    close_pool()    # after app is shut down - closes the pool


app = FastAPI(lifespan=lifespan)
app.include_router(run_router)
app.include_router(epoch_router)

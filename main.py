
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, utils

app = FastAPI(
    title="AI-Enhanced Career Matching API - Top 3 Focus",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers (keeps the exact same paths as before)
app.include_router(analyze.router)
app.include_router(utils.router)

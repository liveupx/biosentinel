"""
BioSentinel — AI-Powered Longitudinal Health Monitoring Platform
FastAPI Application Entry Point

Author: Mohit Chaprana / Liveupx Pvt. Ltd.
License: MIT
Repository: https://github.com/liveupx/biosentinel
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import patients, checkups, predictions, medications, analytics
from src.core.config import settings
from src.core.database import create_tables

app = FastAPI(
    title="BioSentinel API",
    description=(
        "AI-powered longitudinal health monitoring and early disease prediction platform. "
        "Analyzes 3+ years of patient health data to predict cancer and serious diseases "
        "before symptoms appear. Built by Liveupx Pvt. Ltd."
    ),
    version="0.1.0",
    contact={
        "name": "Mohit Chaprana / Liveupx Pvt. Ltd.",
        "url": "https://liveupx.com",
        "email": "biosentinel@liveupx.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/liveupx/biosentinel/blob/main/LICENSE",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await create_tables()


@app.get("/", tags=["Health"])
async def root():
    """Platform status and version."""
    return {
        "name": "BioSentinel",
        "version": "0.1.0",
        "status": "operational",
        "description": "AI-powered early disease detection platform",
        "docs": "/docs",
        "github": "https://github.com/liveupx/biosentinel",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """API health check endpoint."""
    return JSONResponse({"status": "healthy", "api": "v1"})


# Register routers
app.include_router(patients.router, prefix="/api/v1/patients", tags=["Patients"])
app.include_router(checkups.router, prefix="/api/v1/checkups", tags=["Checkups"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(medications.router, prefix="/api/v1/medications", tags=["Medications"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

"""Production FastAPI REST Backend for ScamShield (Phase 23).

Exposes high-throughput, async REST API endpoints for real-time scam detection,
category classification, risk scoring, and dual-layer explainability.

Endpoints:
- GET  /health      : System diagnostics, model status, and feature dimensions.
- POST /predict     : Fast binary threat verdict and calibrated risk score.
- POST /category    : Multi-class 6-class scam taxonomy categorization.
- POST /explain     : Quantitative SHAP attributions and multi-pillar evidence narrative.
- POST /scan        : Complete end-to-end multimodal scan (Verdict + Category + Risk + Explainability).
- POST /scan-batch  : Bulk batch message scanning for enterprise integration.

Documentation:
- Interactive Swagger UI: /docs
- ReDoc Documentation:   /redoc
"""

from __future__ import annotations

import datetime
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline.scamshield_pipeline import ScamShieldPipeline
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------
class MessageScanRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Raw message text to analyze (SMS, WhatsApp, email, or chat).",
        examples=["URGENT: Delhi Police CBI arrest warrant issued against your Aadhaar. Connect to video call http://192.168.1.1/cbi immediately."],
    )
    url: str | None = Field(
        default=None,
        description="Optional explicit URL link associated with the communication.",
        examples=["http://192.168.1.1/cbi"],
    )


class BatchScanRequest(BaseModel):
    messages: list[MessageScanRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of messages to scan in a single batch.",
    )


class VerdictResponse(BaseModel):
    is_threat: bool
    threat_probability: float
    decision_value: float
    threat_level: str
    primary_directive: str


class CategoryResponse(BaseModel):
    predicted_category: str
    category_title: str
    confidence: float
    category_probabilities: dict[str, float]


class RiskAssessmentResponse(BaseModel):
    risk_score: float
    risk_tier: str
    badge_color: str
    confidence_level: str
    action_checklist: list[str]


class SHAPDriver(BaseModel):
    feature: str
    modality: str
    value: float
    impact: float


class UnifiedAssessmentResponse(BaseModel):
    risk_score: float
    risk_level: str
    prediction: str
    category: str
    category_title: str
    confidence: float
    reasons: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    recommended_action: list[str] = Field(default_factory=list)


class ExplanationResponse(BaseModel):
    executive_summary: str
    reasons: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    evidence_pillars: dict[str, Any]
    recommendations: list[str]
    top_threat_drivers: list[SHAPDriver]
    top_benign_drivers: list[SHAPDriver]


class ScanResponse(BaseModel):
    status: str
    timestamp: str
    latency_ms: float
    input: dict[str, Any]
    unified_assessment: UnifiedAssessmentResponse | None = None
    verdict: VerdictResponse
    category: CategoryResponse
    risk_assessment: RiskAssessmentResponse
    explanation: ExplanationResponse


class HealthResponse(BaseModel):
    status: str
    pipeline_name: str
    version: str
    initialized_at: str
    models_loaded: dict[str, Any]
    hardware: dict[str, Any]


# ---------------------------------------------------------------------------
# Lifespan Context Manager (Startup / Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on startup."""
    logger.info("Initializing ScamShield Pipeline for FastAPI service...")
    try:
        pipeline = ScamShieldPipeline.get_instance()
        app.state.pipeline = pipeline
        logger.info("ScamShield Pipeline successfully pre-warmed and cached.")
    except Exception as e:
        logger.error(f"Failed to initialize ScamShield Pipeline: {e}")
        raise e
    yield
    logger.info("Shutting down ScamShield FastAPI service.")


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ScamShield AI REST API",
    description=(
        "Explainable Multimodal AI for Phishing, Extortion, and Digital Scam Detection "
        "Using Text, URL, and Intent Analysis.\n\n"
        "Key Capabilities:\n"
        "- Sub-20ms Multimodal Threat Inference across 15,028 features\n"
        "- 6-Class Indian Scam Taxonomy Classification (Rule 18)\n"
        "- Calibrated 0–100 Risk Scoring across 4 Severity Tiers\n"
        "- Exact Linear SHAP Feature Attributions and Plain-Language Evidence Narratives\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for web apps, dashboards, and Chrome extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/",
    tags=["General"],
    summary="Root Welcome & Documentation Link",
)
def root(request: Request):
    accept = request.headers.get("accept", "")
    index_file = PROJECT_ROOT / "frontend" / "index.html"
    if "text/html" in accept and "application/json" not in accept and index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "ScamShield Multimodal AI REST API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "web_interface": "/app/",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="System Health & Diagnostic Status",
)
def health_check():
    """Returns runtime pipeline health, loaded model types, and feature dimensions."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    return pipeline.get_health()


@app.post(
    "/predict",
    response_model=dict[str, Any],
    tags=["Inference"],
    summary="Fast Binary Threat Verdict & Risk Assessment",
)
def predict_threat(payload: MessageScanRequest):
    """Executes fast binary threat detection and calibrated risk scoring."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    res = pipeline.scan(payload.text, url=payload.url)
    if res["status"] != "success":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return {
        "status": "success",
        "latency_ms": res["latency_ms"],
        "verdict": res["verdict"],
        "risk_assessment": res["risk_assessment"],
    }


@app.post(
    "/category",
    response_model=CategoryResponse,
    tags=["Inference"],
    summary="Scam Taxonomy Multi-Class Categorization",
)
def categorize_message(payload: MessageScanRequest):
    """Classifies communication into the 6-class Indian scam taxonomy."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    res = pipeline.scan(payload.text, url=payload.url)
    if res["status"] != "success":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res["category"]


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Dual-Layer Quantitative & Qualitative Explanation",
)
def explain_threat(payload: MessageScanRequest):
    """Returns exact local SHAP feature attributions and multi-pillar plain-language narrative."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    res = pipeline.scan(payload.text, url=payload.url)
    if res["status"] != "success":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res["explanation"]


@app.post(
    "/scan",
    response_model=ScanResponse,
    tags=["Inference"],
    summary="Master End-to-End Scan (Full Multimodal Assessment)",
)
def scan_message(payload: MessageScanRequest):
    """Executes the complete ScamShield pipeline: Verdict, Category, Risk, and Explainability."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    res = pipeline.scan(payload.text, url=payload.url)
    if res["status"] != "success":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=res.get("message"))
    return res


@app.post(
    "/scan-batch",
    response_model=list[ScanResponse],
    tags=["Inference"],
    summary="Enterprise Bulk Batch Scanner",
)
def scan_batch_messages(payload: BatchScanRequest):
    """Scans multiple messages in a single batch request for enterprise queue pipelines."""
    pipeline: ScamShieldPipeline = app.state.pipeline
    batch_items = [{"text": m.text, "url": m.url} for m in payload.messages]
    results = pipeline.scan_batch(batch_items)
    return results


# ---------------------------------------------------------------------------
# Static File Mounting for Web Frontend & Empirical Visualizations
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "reports"

if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


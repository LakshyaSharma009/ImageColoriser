"""Pydantic response models for the REST API."""
from typing import List, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    label: str
    available: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    models: List[str]
    device: str


class HistoryEntry(BaseModel):
    id: int
    filename: str
    model_id: str
    saturation: float
    width: int
    height: int
    elapsed_s: float
    created_at: str


class HistoryResponse(BaseModel):
    runs: List[HistoryEntry]


class ExperimentEntry(BaseModel):
    id: int
    created_at: str
    model: str
    dataset: str
    image_count: int
    saturation: float
    device: str
    psnr_mean: Optional[float] = None
    ssim_mean: Optional[float] = None
    lpips_mean: Optional[float] = None
    latency_mean_seconds: Optional[float] = None
    latency_median_seconds: Optional[float] = None
    latency_p95_seconds: Optional[float] = None
    git_commit: Optional[str] = None


class ExperimentsResponse(BaseModel):
    experiments: List[ExperimentEntry]

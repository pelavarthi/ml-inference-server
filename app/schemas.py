import math

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    model: str = Field(min_length=1)
    features: list[float] = Field(min_length=1, max_length=1024)

    @field_validator("model")
    @classmethod
    def name_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("model name must not be blank")
        return value.strip()

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, value):
        if not all(math.isfinite(x) for x in value):
            raise ValueError("features must be finite numbers")
        return value


class PredictResponse(BaseModel):
    model: str
    version: str
    prediction: float


class JobResponse(BaseModel):
    job_id: str
    model: str
    status: str
    prediction: float | None = None
    error: str | None = None

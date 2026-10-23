from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model: str
    features: list[float] = Field(min_length=1)


class PredictResponse(BaseModel):
    model: str
    version: str
    prediction: float

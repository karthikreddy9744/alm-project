from pydantic import BaseModel, Field
from typing import List

class SemanticSceneObject(BaseModel):
    internal_reasoning: str = Field(default="", description="Step-by-step internal reasoning.")
    human_oriented_summary: str = Field(default="Unknown Situation", description="A highly empathetic, descriptive human paragraph explaining the situation.")
    primary_situation: str = Field(default="Unknown", description="Extremely concise title (max 7 words).")
    likely_environment: str = Field(default="Unknown", description="Where this is happening.")
    actors: List[str] = Field(default_factory=list, description="People or entities inferred to be present.")
    human_goals: List[str] = Field(default_factory=list, description="Probable goals of the actors.")
    supporting_evidence: str = Field(default="", description="Evidence supporting the primary interpretation.")
    alternative_interpretation: str = Field(default="None", description="Alternative possible situation.")
    missing_evidence: str = Field(default="", description="What we don't know but would help clarify.")
    projection: str = Field(default="Unknown", description="Hints about what might happen next based on evidence.")
    confidence: float = Field(default=0.5, description="0.0 to 1.0 confidence in the primary situation.")

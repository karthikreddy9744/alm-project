from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class RelationshipToHypothesis(str, Enum):
    PRIMARY_SUPPORT = "PrimarySupport"
    SECONDARY_SUPPORT = "SecondarySupport"
    CONTRADICTORY = "Contradictory"
    CONTEXTUAL = "Contextual"
    INCIDENTAL = "Incidental"
    LOW_CONFIDENCE = "LowConfidence"

class Influence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    IGNORED = "Ignored"

class AgreementLevel(str, Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    CONTRADICTORY = "Contradictory"

class VerificationStatus(str, Enum):
    STRONGLY_SUPPORTED = "Strongly Supported"
    MODERATELY_SUPPORTED = "Moderately Supported"
    WEAKLY_SUPPORTED = "Weakly Supported"
    CONTRADICTED = "Contradicted"
    INCONCLUSIVE = "Inconclusive"

class DominantModality(str, Enum):
    SPEECH = "Speech"
    ENVIRONMENT = "Environment"
    BALANCED = "Balanced"

class SourceType(str, Enum):
    REAL_WORLD = "RealWorld"
    BROADCAST = "Broadcast"
    MEDIA_PRODUCTION = "MediaProduction"
    PODCAST = "Podcast"
    USER_RECORDING = "UserRecording"
    SURVEILLANCE = "Surveillance"
    SYNTHETIC = "Synthetic"
    UNKNOWN = "Unknown"

class RepresentationType(str, Enum):
    LITERAL = "Literal"
    REENACTMENT = "Reenactment"
    FICTION = "Fiction"
    REPORTING = "Reporting"
    EDUCATIONAL = "Educational"
    ENTERTAINMENT = "Entertainment"
    DOCUMENTARY = "Documentary"
    PERFORMANCE = "Performance"
    UNKNOWN = "Unknown"

class ProvenanceReliability(str, Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    UNKNOWN = "Unknown"

class AudioProvenanceReasoning(BaseModel):
    source_type: SourceType
    representation_type: RepresentationType
    confidence: float = Field(description="Confidence propagated from perception uncertainty.")
    provenance_reliability: ProvenanceReliability
    supporting_evidence: List[str] = Field(description="Evidence supporting this provenance.")
    remaining_uncertainty: str = Field(description="What is still unknown about the source?")

class SpeechUnderstanding(BaseModel):
    summary: str = Field(description="Brief summary of what was spoken.")
    topic: str = Field(description="The semantic topic.")
    speaker_intent: str = Field(description="Why are they speaking?")
    emotional_tone: str = Field(description="Emotional state of the speaker.")
    confidence: float = Field(description="0.0 to 1.0 confidence in the speech understanding.")

class AuditoryObservation(BaseModel):
    id: str = Field(description="Unique ID for this observation (e.g., obs_01).")
    sound: str = Field(description="The detected sound event.")
    evidence_source: str = Field(description="Source model (e.g., HTS-AT, CLAP).")
    start_time: float = Field(description="Start time in seconds.")
    end_time: float = Field(description="End time in seconds.")
    detection_confidence: float = Field(description="Original confidence from the perception model.")
    relationship_to_hypothesis: RelationshipToHypothesis
    influence: Influence
    used_in_final_reasoning: bool = Field(description="Did this materially affect the final interpretation?")
    justification: str = Field(description="Extremely short reason why.")

class CrossModalAssessment(BaseModel):
    agreement_level: AgreementLevel
    verification_status: VerificationStatus
    dominant_modality: DominantModality
    major_supports: List[str] = Field(default_factory=list, description="List of observation IDs (e.g., ['obs_01', 'obs_03'])")
    major_conflicts: List[str] = Field(default_factory=list, description="List of observation IDs (e.g., ['obs_02'])")
    remaining_uncertainty: str = Field(description="What is still unknown?")
    overall_assessment: str = Field(description="1-2 sentence summary of cross-modal consistency.")

class SemanticSceneObject(BaseModel):
    speech_understanding: SpeechUnderstanding
    auditory_observations: List[AuditoryObservation]
    audio_provenance_reasoning: AudioProvenanceReasoning
    cross_modal_assessment: CrossModalAssessment
    primary_situation: str = Field(description="Extremely concise title (max 7 words).")
    environmental_context: str = Field(description="The synthesized physical environment.")
    actors: List[str] = Field(default_factory=list, description="People or entities inferred to be present.")
    human_goals: List[str] = Field(default_factory=list, description="Probable goals of the actors.")
    alternative_hypotheses: List[str] = Field(default_factory=list, description="1-3 plausible alternative situations.")
    missing_evidence: List[str] = Field(default_factory=list, description="Evidence that would increase certainty.")
    likely_next_state: str = Field(description="Estimation of the immediate next state.")
    interpretation_confidence: float = Field(description="Final overall confidence (0.0 to 1.0).")
    human_oriented_summary: str = Field(description="A highly empathetic, descriptive human paragraph explaining the situation.")

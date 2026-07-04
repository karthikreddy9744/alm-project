from typing import List, Optional
from reasoning_engine.awm.models import HierarchicalConfidence, UncertaintyObject, Trajectory
from reasoning_engine.awm.world_model import AuditoryWorldModel

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.4
HIGH_CONFIDENCE_THRESHOLD = 0.8
MAX_UNCERTAINTY_PENALTY = 0.2

# -----------------------------------------------------------------------------
# Confidence & Weighting Calculations
# -----------------------------------------------------------------------------

def evaluate_evidence_confidence(p_val: float) -> HierarchicalConfidence:
    """Evaluates raw probability and returns a structured HierarchicalConfidence."""
    conf = HierarchicalConfidence()
    conf.fusion = p_val
    if p_val < CONFIDENCE_THRESHOLD:
        uncertainty = UncertaintyObject(reason="Signal probability below threshold", is_conflict=False)
        conf.is_uncertain = True
        conf.uncertainty_metadata = uncertainty
    else:
        conf.is_uncertain = False
    return conf

def calculate_support_score(confidences: List[HierarchicalConfidence]) -> float:
    """Calculates the aggregated support score for a hypothesis based on its evidence."""
    if not confidences:
        return 0.0
    
    # Simple average for determinism; penalties applied for uncertainty
    base_score = sum(c.p_val for c in confidences) / len(confidences)
    
    uncertainty_penalty = sum(MAX_UNCERTAINTY_PENALTY for c in confidences if c.is_uncertain)
    final_score = max(0.0, base_score - uncertainty_penalty)
    return round(final_score, 4)

def check_for_contradiction(trajectory_a: Trajectory, trajectory_b: Trajectory) -> float:
    """
    Returns a penalty if two trajectories logically contradict each other
    (e.g., an approaching event mapped to a receding primitive).
    """
    if trajectory_a == Trajectory.APPROACHING and trajectory_b == Trajectory.RECEDING:
        return 1.0  # Maximum contradiction
    if trajectory_a == Trajectory.STATIC and trajectory_b in [Trajectory.APPROACHING, Trajectory.RECEDING]:
        return 0.5  # Partial contradiction
    return 0.0

# -----------------------------------------------------------------------------
# Trajectory Helpers
# -----------------------------------------------------------------------------

def calculate_trajectory(current_vol: float, previous_vol: Optional[float]) -> Trajectory:
    """Determines acoustic trajectory deterministically based on delta volume/salience."""
    if previous_vol is None:
        return Trajectory.STATIC
    
    delta = current_vol - previous_vol
    
    if delta > 0.1:
        return Trajectory.APPROACHING
    elif delta < -0.1:
        return Trajectory.RECEDING
    else:
        return Trajectory.STATIC

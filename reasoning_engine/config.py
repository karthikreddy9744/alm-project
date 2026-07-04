import re

# ==========================================
# Belief State Engine (BSE) Configuration
# ==========================================
BSE_EXPIRATION_TIME_SECONDS = 300.0  # Beliefs inactive for 5 mins are removed
BSE_WEAKEN_TIME_SECONDS = 15.0       # Beliefs begin weakening after 15 seconds without updates
BSE_MAX_ACTIVE_BELIEFS = 1000
BSE_REINFORCEMENT_BOOST = 0.1        # How much confidence to add per consistent reinforcement
BSE_CONTRADICTION_PENALTY = 0.2      # How much confidence to remove per contradiction
BSE_WEAKENING_RATE = 0.05            # Confidence lost per weaken cycle
BSE_MIN_CONFIDENCE_THRESHOLD = 0.1   # Below this, beliefs go INACTIVE

# ==========================================
# Hypothesis Reasoning Engine (HRE) Configuration
# ==========================================
HRE_EXPIRATION_TIME_SECONDS = 300.0  # Hypotheses older than 5 mins without update are expired
HRE_MAX_ACTIVE_HYPOTHESES = 100
HRE_REJECTION_THRESHOLD = 0.1        # Hypotheses with confidence/plausibility below this are REJECTED
HRE_MERGE_SIMILARITY_THRESHOLD = 0.8 # Score threshold to merge hypotheses
HRE_REINFORCEMENT_BOOST = 0.15       # Confidence boost for confirming evidence
HRE_CONTRADICTION_PENALTY = 0.25     # Confidence penalty for contradictory evidence

# ==========================================
# World State Estimation Engine (WSE) Configuration
# ==========================================
# Dictates how strongly the current state resists changing to a new dominant state.
# Requires the new state's score to beat the old state's score by this margin.
WSE_STATE_TRANSITION_MOMENTUM = 0.15 
# Thresholds for when secondary hypotheses are considered ambiguous vs safely dominated
WSE_AMBIGUITY_THRESHOLD = 0.2
WSE_STATE_EXPIRATION_SECONDS = 60.0

# ==========================================
# Situation Projection Engine (SPE) Configuration
# ==========================================
SPE_TRANSITION_RULES = [
    (r"(?i)\bargument\b", "Possible Distress"),
    (r"(?i)\bconversation\b", "Argument"),
    (r"(?i)\bdistress\b", "Possible Emergency"),
    (r"(?i)\bnormal traffic\b", "Cleared Road"),
    (r"(?i)\btraffic jam\b", "Normal Traffic"),
    (r"(?i)\btraffic\b", "Traffic Jam"),
    (r"(?i)\bemergency\b", "Resolution")
]
SPE_HIGH_RISK_KEYWORDS = ["emergency", "distress", "fire", "weapon", "alarm", "shouting", "running"]
SPE_MODERATE_RISK_KEYWORDS = ["argument", "traffic jam", "crowd", "raised voices"]
SPE_PROJECTION_EXPIRATION_SECONDS = 120.0

# ==========================================
# Transparent Reasoning Engine (TRE) Configuration
# ==========================================
TRE_HIGH_CONFIDENCE_THRESHOLD = 0.8
TRE_HIGH_UNCERTAINTY_THRESHOLD = 0.6
TRE_HIGH_CONTRADICTION_THRESHOLD = 0.3
TRE_EXP_STRONG_SUPPORT = "Strong supporting evidence."
TRE_EXP_WEAK_SUPPORT = "Weak supporting evidence."
TRE_EXP_HIGH_UNCERTAINTY = "Significant missing context or ambiguity."
TRE_EXP_LOW_UNCERTAINTY = "Context is relatively clear."
TRE_EXP_HEAVY_CONTRADICTION = "Chosen despite heavy contradiction."
TRE_EXP_LITTLE_CONTRADICTION = "Little to no contradictory evidence."

# ==========================================
# Situation Intelligence Renderer (SIR) Configuration
# ==========================================
SIR_MAX_COMPACT_LENGTH = 140
SIR_MAX_EMERGENCY_SUMMARY_LENGTH = 250

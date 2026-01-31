from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

# --- Core Models (Same as before) ---
class RetrievedChunk(BaseModel):
    content: str
    source_metadata: str
    relevance_reason: str

class AtomicContentUnit(BaseModel):
    acu_type: Literal["definition", "concept", "formula", "unit", "logical_step", "diagram_element"]
    content: str
    max_weight: float
    raw_student_text: Optional[str] = None
    reasoning: Optional[str] = None

class EvaluationLogic(BaseModel):
    question_intent: str
    assumptions: List[str]
    strict_policies: List[str]
    flexibility_strategy: str

class Rubric(BaseModel):
    sub_class: str
    subject: str
    chapter: str
    total_possible_score: float
    base_retrieved_context: List[RetrievedChunk]
    student_retrieved_context: List[RetrievedChunk]
    base_answer_decomposition: List[AtomicContentUnit]
    student_answer_decomposition: List[AtomicContentUnit]
    logic_guidelines: EvaluationLogic
    alternative_valid_points: List[AtomicContentUnit] = []

# app/schema.py

# ... (Keep Rubric and other models the same) ...

class ClaimVerdict(BaseModel):
    student_claim: str = Field(..., description="The specific distinct point extracted from the student's answer.")
    rubric_item_matched: str = Field(..., description="The exact 'Expected Fact' or 'Alternative Point' from the rubric this claim matches.")
    status: Literal["Full Match", "Partial Match", "Alternative Correct", "Incorrect"]
    marks_awarded: float
    reasoning: str = Field(..., description="Detailed explanation: Why does this claim match (or not match) the rubric item? Quote the student text.")

class GradingReport(BaseModel):
    student_id: str = "student_01"
    
    # --- NEW COMPONENT ---
    scoring_logic_summary: str = Field(..., description="A step-by-step text summary of how the score was calculated (e.g., 'Base marks: 2.0, Alternative marks: +1.0, Deductions: -0.5 for units').")
    # ---------------------
    
    final_score: float
    max_possible: float
    confidence_score: float
    verdicts: List[ClaimVerdict]
    policy_deductions: List[Dict[str, str]]
    hitl_flag: bool
    feedback_for_student: str
# --- API Request Models ---

class RubricRequest(BaseModel):
    """Input for Step 1: Generating the Rubric"""
    question: str
    base_ans: str
    student_ans: str = Field(..., description="Used for flexibility check only.")
    total_score: float
    class_level: str = "10"
    subject: str = "Science"
    chapter: str = "General"

class EvaluationRequest(BaseModel):
    """Input for Step 2: Evaluation. PASS THE FULL RUBRIC HERE."""
    student_ans: str
    rubric: Rubric
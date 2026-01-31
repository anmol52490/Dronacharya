from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class RetrievedChunk(BaseModel):
    content: str = Field(..., description="The actual text from the textbook/notes.")
    source_metadata: str = Field(..., description="Chapter, section, or page number.")
    relevance_reason: str = Field(..., description="Why was this chunk retrieved for this specific answer?")

class AtomicContentUnit(BaseModel):
    acu_type: Literal["definition", "concept", "formula", "unit", "logical_step", "diagram_element"]
    content: str = Field(..., description="The core factual statement or requirement.")
    max_weight: float = Field(..., description="Maximum marks possible for this unit.")
    
    # For Student ACUs, we add evaluation placeholders
    raw_student_text: Optional[str] = Field(None, description="The specific part of student's answer mapped here.")
    alignment_confidence: Optional[float] = Field(None, description="0.0 to 1.0 confidence that student's point matches this ACU.")
    reasoning: Optional[str] = Field(None, description="Justification for the alignment or lack thereof.")

class EvaluationLogic(BaseModel):
    question_intent: str = Field(..., description="The fundamental concept the teacher is trying to test.")
    assumptions: List[str] = Field(default_factory=list, description="Accepted scientific variations (e.g., gravity = 10m/s^2).")
    strict_policies: List[str] = Field(default_factory=list, description="Non-negotiable rules (e.g., 'must include units', 'zero marks for wrong diagram').")
    flexibility_strategy: str = Field(..., description="How to handle valid knowledge that isn't in the base answer.")

class Rubric(BaseModel):
    # 1. Metadata
    sub_class: str
    subject: str
    chapter: str
    total_possible_score: float
    
    # 2. RAG Context (The Scientific Foundation)
    base_retrieved_context: List[RetrievedChunk] = Field(..., description="Chunks retrieved using Question + Base Answer.")
    student_retrieved_context: List[RetrievedChunk] = Field(..., description="Chunks retrieved using Question + Student Answer.")
    
    # 3. The Decomposition (The Logic Map)
    base_answer_decomposition: List[AtomicContentUnit] = Field(..., description="Base answer broken into atomic facts.")
    student_answer_decomposition: List[AtomicContentUnit] = Field(..., description="Student's response parsed into atomic claims.")
    
    # 4. Evaluator Rules
    logic_guidelines: EvaluationLogic
    
    # 5. Semantic Bridge
    alternative_valid_points: List[AtomicContentUnit] = Field(
        default_factory=list, 
        description="Valid facts found in Student's answer + Student RAG that weren't in the Base Answer."
    )
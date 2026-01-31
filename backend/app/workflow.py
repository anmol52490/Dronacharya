import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .schema import Rubric, GradingReport, RubricRequest, ConsensusReport
from .retriever import get_relevant_context

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workflow")

# LLMs
llm_gen = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
llm_eval = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

structured_llm_gen = llm_gen.with_structured_output(Rubric)
structured_llm_eval = llm_eval.with_structured_output(GradingReport)

# --- State Definitions ---
class RubricState(TypedDict):
    inputs: dict
    rag_data: dict
    rubric: Optional[Rubric]

class EvalState(TypedDict):
    inputs: dict
    rubric: Optional[Rubric]
    # Change this from GradingReport to ConsensusReport
    final_report: Optional[ConsensusReport]

# --- NODES ---

def retrieval_node(state: RubricState):
    logger.info("🔍 Retrieving Context...")
    q = state["inputs"]["question"]
    base_ans = state["inputs"]["base_ans"]
    student_ans = state["inputs"]["student_ans"]
    
    base_context = get_relevant_context(f"{q} {base_ans}")
    student_context = get_relevant_context(f"{q} {student_ans}")
    
    return {"rag_data": {"base": base_context, "student": student_context}}

def rubric_generator_node(state: RubricState):
    logger.info("🧠 Generating Rubric...")
    inputs = state["inputs"]
    rag = state["rag_data"]

    prompt = f"""
    ROLE: You are an expert Lead Teacher creating a precise grading rubric.
    
    TASK:
    Analyze the provided question, perfect answer, and student response. Use the 
    Textbook Context as the ultimate scientific truth to verify all claims.
    
    INPUT DATA:
    - Question: {inputs['question']}
    - Perfect Answer: {inputs['base_ans']}
    - Student's Written Answer: {inputs['student_ans']}
    - Textbook Context (Standard): {[c.content for c in rag['base']]}
    - Textbook Context (Student-specific): {[c.content for c in rag['student']]}
    
    INSTRUCTIONS:
    1. THE BREAKDOWN: Break the 'Perfect Answer' into the smallest individual facts or steps (Information Bits). 
       Assign a portion of the total {inputs['total_score']} marks to each bit.
    
    2. STUDENT ANALYSIS: Identify and list every individual claim the student made in their written answer.
    
    3. FLEXIBILITY CHECK: Look at the 'Student-specific Textbook Context'. If the student mentioned a 
       fact that is correct according to the textbook but NOT in the perfect answer, list it as an 'alternative_valid_point'.
    
    4. INTENT & POLICY: 
       - Identify the core concept the student must prove they understand (question_intent).
       - Set 'assumptions' (e.g., acceptable synonyms or rounding).
       - Set 'strict_policies' (e.g., marks deducted if units are missing).
    
    5. MAPPING: For the student's claims, explain in the 'reasoning' field how well they match 
       the perfect answer or the textbook notes.
    """


    rubric = structured_llm_gen.invoke(prompt)
    return {"rubric": rubric}
# app/workflow.py




def evaluator_node(state: EvalState):
    logger.info("⚖️ Evaluating (3 Parallel Runs)...")    
    # 1. Recover the Rubric Object from the dictionary
    rubric_dict = state["inputs"]["rubric"]
    rubric = Rubric(**rubric_dict)
    
    student_ans = state["inputs"]["student_ans"]

    # --- THE METICULOUS PROMPT ---
    eval_prompt = f"""
    ROLE: You are a Lead Academic Examiner known for extreme precision.
    
    TASK: Grade the STUDENT ANSWER strictly against the provided MASTER RUBRIC. 
    You must justify every fraction of a mark awarded or deducted.
    
    --- 1. THE DATA ---
    STUDENT ANSWER: "{student_ans}"
    
    MASTER RUBRIC:
    A. ESSENTIAL FACTS (Max Score per item shown): 
       {rubric.base_answer_decomposition}
       
    B. ALTERNATIVE ALLOWED FACTS (Credit these if Essential Facts are missing): 
       {rubric.alternative_valid_points}
       
    C. NEGATIVE MARKING POLICIES (Strict Deductions): 
       {rubric.logic_guidelines.strict_policies}
       
    D. TOTAL MAX SCORE: {rubric.total_possible_score}
    
    --- 2. THE STRICT GRADING PROCEDURE ---
    
    STEP 1: CLAIM EXTRACTION & MATCHING
    - Identify every distinct scientific claim in the Student Answer.
    - Compare each claim against the 'ESSENTIAL FACTS'.
      - IF MATCH: Award full marks. Status = "Full Match". Quote the Rubric item matched.
      - IF NO MATCH: Check 'ALTERNATIVE ALLOWED FACTS'.
        - IF MATCH: Award full marks. Status = "Alternative Correct".
      - IF NO MATCH IN EITHER: Award 0 marks. Status = "Incorrect".
      
    STEP 2: PARTIAL CREDIT CHECK
    - If a student's claim is vaguely correct but missing keywords (e.g., "It pushes" instead of "Force applied"), check the Rubric's 'assumptions'.
    - If acceptable, award marks. If too vague, mark "Partial Match" and give 50%.
    
    STEP 3: POLICY AUDIT (DEDUCTIONS)
    - Scan the entire answer against 'NEGATIVE MARKING POLICIES'.
    - If a rule is violated (e.g., "No units"), apply the deduction immediately.
    - Record the exact policy text and the amount deducted.
    
    STEP 4: FINAL CALCULATION
    - (Sum of Marks from Claims) - (Total Deductions) = Final Score.
    - CAP the score: It cannot exceed {rubric.total_possible_score} or be less than 0.
    
    STEP 5: GENERATE REPORT
    - 'scoring_logic_summary': Write a plain-text summary of the math (e.g., "Student earned 2.0 from facts, got 1.0 bonus for alternative, lost 0.5 for missing units.").
    - 'verdicts': List the judgment for every claim.
    - 'feedback_for_student': Constructive feedback based on what was missing.
    """

    # Run the evaluation (Parallel Simulation)
    # We run 3 times to ensure the logic holds up across different inference paths
    reports = [structured_llm_eval.invoke(eval_prompt) for _ in range(3)]

    # --- CONSENSUS LOGIC ---
    scores = [r.final_score for r in reports]
    avg_score = sum(scores) / 3
    
    # Calculate Variance
    variance = max(scores) - min(scores)
    is_flagged = variance > 2

    # Choose the report that is closest to the average (most representative)
    consensus = ConsensusReport(
        consensus_score=round(avg_score, 2),
        score_variance=round(variance, 2),
        hitl_flag=is_flagged,
        individual_runs=reports  # <--- WE SAVE ALL 3 HERE
    )

    return {"final_report": consensus}

# --- 1. Rubric Graph ---
rubric_workflow = StateGraph(RubricState)
rubric_workflow.add_node("retrieve", retrieval_node)
rubric_workflow.add_node("generate", rubric_generator_node)
rubric_workflow.set_entry_point("retrieve")
rubric_workflow.add_edge("retrieve", "generate")
rubric_workflow.add_edge("generate", END)
rubric_app = rubric_workflow.compile()

# --- 2. Eval Graph (Direct) ---
eval_workflow = StateGraph(EvalState)
eval_workflow.add_node("evaluate", evaluator_node)
eval_workflow.set_entry_point("evaluate")
eval_workflow.add_edge("evaluate", END)
eval_app = eval_workflow.compile()
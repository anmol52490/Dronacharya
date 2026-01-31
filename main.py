import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import Rubric
from retriever import get_relevant_context

# 1. State Definition
class GraphState(TypedDict):
    inputs: dict  # {question, student_ans, base_ans, total_score, class, subject, chapter}
    rag_data: dict # {base_context, student_context}
    rubric: Optional[Rubric]

# 2. Initialize LLM with structured output using your existing Rubric schema
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
structured_llm = llm.with_structured_output(Rubric)

# 3. Nodes
def retrieval_node(state: GraphState):
    """Fetches textbook context for both the teacher's and student's logic."""
    q = state['inputs']['question']
    # Use the base answer to find 'Standard' truth
    base_context = get_relevant_context(f"{q} {state['inputs']['base_ans']}")
    # Use the student answer to find 'Alternative' or 'Expanded' truth
    student_context = get_relevant_context(f"{q} {state['inputs']['student_ans']}")
    
    return {"rag_data": {"base": base_context, "student": student_context}}

def rubric_generator_node(state: GraphState):
    """The Brain: Acts as an expert teacher to build the grading map."""
    inputs = state['inputs']
    rag = state['rag_data']
    
    # We maintain the schema names (AtomicContentUnit) but explain them 
    # simply in the prompt as 'Information Bits' or 'Facts'.
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
    
    result = structured_llm.invoke(prompt)
    return {"rubric": result}

# 4. Build LangGraph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve_context", retrieval_node)
workflow.add_node("generate_rubric", rubric_generator_node)

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_rubric")
workflow.add_edge("generate_rubric", END)

app = workflow.compile()

# --- Execution Example ---
if __name__ == "__main__":
    example_input = {
        "question": "Explain how plants make food.",
        "base_ans": "Plants use photosynthesis to convert sunlight, CO2, and water into glucose and oxygen.",
        "student_ans": "They use light energy from the sun to turn water and carbon dioxide into sugar. They also release O2 as a byproduct.",
        "total_score": 3.0,
        "class": "10", 
        "subject": "Biology", 
        "chapter": "Life Processes"
    }

    final_state = app.invoke({"inputs": example_input})
    
    if final_state['rubric']:
        # Convert to dict first, then use standard json.dumps for pretty printing
        rubric_data = final_state['rubric'].model_dump()
        import json
        print(json.dumps(rubric_data, indent=2))
    else:
        print("Rubric generation failed.")
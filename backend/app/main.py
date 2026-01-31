from fastapi import FastAPI, HTTPException
from .schema import RubricRequest, EvaluationRequest, GradingReport, Rubric
from .workflow import rubric_app, eval_app

app = FastAPI(title="DrnaAI Minimalist Engine", version="3.0")

@app.post("/generate-rubric", response_model=Rubric)
def generate_rubric(request: RubricRequest):
    """
    Step 1: Returns a full Rubric JSON. 
    You must copy this JSON to use in Step 2.
    """
    try:
        input_data = request.model_dump()
        result = rubric_app.invoke({"inputs": input_data})
        return result["rubric"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=GradingReport)
def evaluate_student(request: EvaluationRequest):
    """
    Step 2: Takes { "student_ans": "...", "rubric": { ... } }
    Returns the grade.
    """
    try:
        # Pydantic handles the parsing of the 'rubric' field automatically
        input_data = request.model_dump()
        
        result = eval_app.invoke({"inputs": input_data})
        
        if not result.get("final_report"):
            raise HTTPException(status_code=500, detail="Evaluation failed.")
            
        return result["final_report"]
        
    except Exception as e:
        print(f"EVAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
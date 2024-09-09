from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import pandas as pd
from pydantic import BaseModel
import json
import logging
from Resume_shortlist_system import ResumeShortlistSystem

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Shortlisting System")

RS_system = ResumeShortlistSystem()

class CustomWeightage(BaseModel):
    skills: float
    work_experience: float
    projects: float
    education: float
    certifications: float

async def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
    finally:
        await upload_file.close()

@app.post("/process_resumes/")
async def process_resumes(
    job_description: UploadFile = File(...),
    resumes: List[UploadFile] = File(...),
    weightage: Optional[str] = Form("Default"),
    custom_weightage: Optional[str] = Form(None)
):
    try:
        logger.info("Starting resume processing")
        RS_system.clear_previous_data

        job_id = os.path.splitext(job_description.filename)[0]
        logger.info(f"job description: {job_id}")
        # job_file_path = f"temp_{job_description.filename}"
        # logger.info(f"job description path: {job_file_path}")
        # await save_upload_file(job_description, job_file_path)
        job_store = RS_system.load_document(job_description, "job_store")
        # os.remove(job_file_path)
        logger.info(f"Processed job description: {job_id}")

        results = []
        for resume in resumes:
            resume_id = os.path.splitext(resume.filename)[0]
            resume_id = RS_system.sanitize_filename(resume_id)
            # resume_file_path = f"temp_{resume_id}"
            # await save_upload_file(resume, resume_file_path)
            resume_store = RS_system.load_document(resume, f"resume_stores/{resume_id}")
            match_result = RS_system.match_resumes(job_store, resume_store, resume_id, job_id)
            if match_result is None:
                logger.warning(f"Failed to process resume: {resume_id}")
                continue
            match_result = RS_system.get_total_duration(match_result)
            results.append(match_result)
            # os.remove(resume_file_path)
            logger.info(f"Processed resume: {resume_id}")

        if not results:
            raise HTTPException(status_code=400, detail="No valid results were generated from the resumes")

        custom_weights = None
        if weightage == "Custom" and custom_weightage:
            try:
                custom_weights = json.loads(custom_weightage)
                custom_weights = CustomWeightage(**custom_weights)
                is_valid, error_message = RS_system.validate_weights(custom_weights.dict())
                if not is_valid:
                    raise HTTPException(status_code=400, detail=error_message)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON for custom_weightage")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        logger.info("Generating final DataFrame")
        df = RS_system.final_df(results, custom_weights=custom_weights.dict() if custom_weights else None)

        response_data = {
            "job_id": job_id,
            "results": results,
            "summary": df.to_dict(orient="records")
        }

        logger.info("Processing completed successfully")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
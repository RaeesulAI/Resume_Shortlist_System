import os
import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from Resume_shortlist_system import ResumeShortlistSystem

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume Shortlisting System")

RS_system = ResumeShortlistSystem()

# FastAPI endpoint for processing resumes and job descriptions
@app.post("/process_resumes/")
async def process_resumes(
    job_description: UploadFile = File(...),
    resumes: List[UploadFile] = File(...),
    weightage: Optional[str] = Form("Default"),
    skills: Optional[float] = None,
    experience: Optional[float] = None,
    projects: Optional[float] = None,
    education: Optional[float] = None,
    certification: Optional[float] = None
):
    try:
        logger.info("Starting resume processing")
        # RS_system.clear_previous_data

        job_id = os.path.splitext(job_description.filename)[0]
        logger.info(f"job description: {job_id}")
        job_store = RS_system.load_document(job_description, f"job_store/{job_id}")
        logger.info(f"Processed job description: {job_id}")

        results = []
        for resume in resumes:
            resume_id = os.path.splitext(resume.filename)[0]
            resume_id = RS_system.sanitize_filename(resume_id)
            resume_store = RS_system.load_document(resume, f"resume_stores/{resume_id}")
            match_result = RS_system.match_resumes(job_store, resume_store, resume_id, job_id)
            if match_result is None:
                logger.warning(f"Failed to process resume: {resume_id}")
                continue
            match_result = RS_system.get_total_duration(match_result)
            results.append(match_result)
            logger.info(f"Processed resume: {resume_id}")

        if not results:
            raise HTTPException(status_code=400, detail="No valid results were generated from the resumes")

        custom_weights = None
        if weightage == "Custom":
            if None in [skills, experience, projects, education, certification]:
                raise HTTPException(status_code=400, detail="All custom weights must be provided.")
            custom_weightage = {
                "skills": skills,
                "work_experience": experience,
                "projects": projects,
                "education": education,
                "certifications": certification
            }
            is_valid, error_message = RS_system.validate_weights(custom_weightage)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_message)

        logger.info("Generating final DataFrame")
        df = RS_system.final_df(results, custom_weights=custom_weights.dict() if custom_weights else None)

        response_data = {
            "job_id": job_id,
            "results": results,
            "summary": df.to_dict(orient="records"),
            "csv_data": df.to_csv(index=False)
        }

        logger.info("Processing completed successfully")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
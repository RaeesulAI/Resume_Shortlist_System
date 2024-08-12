import json
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Create a custom prompt template
custom_prompt_template = """
You are an experienced Technical Human Resource (HR) Manager and skilled ATS (Applicant Tracking System) scanner with a deep understanding of tech field and ATS functionality.
Your task is to analyze the provided resume (CV) and job description (JD) to determine how well they match in several key areas.
        
First, extract the job position from the job description.
        
Then, you extract the following personal information from the resume:
    1. Candidate's full name
    2. Email address
    3. Contact number

Next, you need to calculate matching percentages for the following categories between resume (CV) and job description (JD):
    1. Skills (Technical and Soft skills)
    2. Work Experience
    3. Projects
    4. Professional Certifications
    5. Education, Achievements, and Extra Curricular Activities

For each category, provide a detailed analysis and a matching percentage.
After analyzing each category, provide a final overall matching percentage by aggregating the individual scores.

For the work experience, follow these rules:
    1. If the resume doesn't mention any work experience or if the candidate has no work experience, set all work experience fields to "No Experience" and the duration to 0.
    2. Calculate the total duration by summing up all individual job durations.
    3. If there are overlapping periods, count them only once.
    4. For any job listed as "present" or "current", calculate the duration up to {current_date}.
    5. Handle various date formats consistently (e.g., "March 2024", "Mar 2024", "2024 March", "2024 Mar").
    6. Express each job duration and the total duration in years and months.
    7. For date formats that include days (e.g., "19th February 2024"), extract only the month and year (e.g., "February 2024").

If any information is not available or cannot be determined, use the phrase "Not Mentioned" for that field.
        
Provide the output as a valid JSON string in the following format:
{{
    "job_position": "extracted job position",

    "personal_info": {{
        "name": "Full Name",
        "email": "email@example.com",
        "contact_number": ["contactnumber1", "contactnumber2"],
    }},
    "skills_matching": {{
        "technical_skills": ["skill1", "skill2", "skill3"],
        "soft_skills": ["soft1", "soft2", "soft3"],
        "skills_percentage": "Skills Matching Percentage",
        "skills_analysis": "Detailed explanation of the matching process and results"
    }},
    "work_experience_matching": {{
        "job_roles": ["role1", "role2"] or ["No Experience"],
        "relevant_job_roles": [
        {{
            "title": "Job Title",
            "company": "Company Name",
            "start_date": "YYYY-MM",
            "end_date": "YYYY-MM or 'present'",
            "duration": {{
                "years": X,
                "months": Y
            }}
        }}
        ] or [],
        "total_duration": {{
            "years": X,
            "months": Y
        }},
        "key_responsibilities": ["responsibility1", "responsibility2", "responsibility3"] or ["No Experience"],
        "experience_percentage": "Work Experience Matching Percentage",
        "experience_analysis": "Detailed explanation of the matching process and results"
    }},
    "projects_matching": {{
        "relevant_projects": ["project1", "project2"],
        "technologies_and_outcomes": "Analysis of technologies used and project outcomes",
        "projects_percentage": "Projects Matching Percentage",
        "projects_analysis": "Detailed explanation of the matching process and results"
    }},
    "education_matching": {{
        "education": ["degree1", "degree2"],
        "achievements": ["achievement1", "achievement2"],
        "extra_activities": ["activity1", "activity2"],
        "education_percentage": "Education, Achievements, and Extra Curricular Activities Matching Percentage",
        "education_analysis": "Detailed explanation of the matching process and results"
    }},
    "professional_certifications_matching": {{
        "certifications": ["cert1", "cert2"],
        "certifications_percentage": "Professional Certifications Matching Percentage",
        "certifications_analysis": "Detailed explanation of the matching process and results"
    }},
    "final_matching": {{
        "final_overall_percentage": "Calculated overall matching percentage",
        "final_analysis": "Detailed explanation of how the final percentage was calculated"
    }}
}}

Job Description:
{job_description}

Resume:
{resume}

Current Date: {current_date}

Analyze the match between the job description and the resume based on the format provided above. 
Ensure the output is a valid JSON string and all fields are consistently included, using "Not Mentioned" for any missing information.
"""

# function for Resume Matching and Result Generation
def match_resumes(job_description_store, resume_store, resume_id, jd_id):
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
    
    job_retriever = job_description_store.as_retriever(search_kwargs={"k": 5})
    resume_retriever = resume_store.as_retriever(search_kwargs={"k": 5})

    current_date = datetime.now().strftime("%B %Y")
    
    job_docs = job_retriever.get_relevant_documents("job description")
    resume_docs = resume_retriever.get_relevant_documents("resume")
    
    job_description = " ".join([doc.page_content for doc in job_docs])
    resume = " ".join([doc.page_content for doc in resume_docs])
    
    match_analysis = model.invoke(custom_prompt_template.format(
        job_description=job_description,
        resume=resume,
        current_date=current_date
    ))

    # Parse the JSON output
    try:
        # Find the start and end of the JSON object
        start = match_analysis.content.find('{')
        end = match_analysis.content.rfind('}') + 1
        json_str = match_analysis.content[start:end]
        analysis_dict = json.loads(json_str)
        analysis_dict['resume_id'] = resume_id
        analysis_dict['job_id'] = jd_id
        return analysis_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Getting an Error for: {resume_id}")
        return None

# function Calculates the duration between two dates
def calculate_duration(start_date, end_date):
    if start_date == "No Experience" or end_date == "No Experience":
        return {"years": 0, "months": 0}
    
    if end_date.lower() == 'present':
        end_date = datetime.now()
    else:
        end_date = parse_date(end_date)
    
    start_date = parse_date(start_date)
    
    if start_date and end_date:
        duration = relativedelta(end_date, start_date)
        return {
            "years": duration.years,
            "months": duration.months
        }
    return {"years": 0, "months": 0}

# function for Attempts to parse various date formats.
def parse_date(date_str):
    date_formats = [
        "%Y-%m",
        "%B %Y",
        "%b %Y",
        "%Y %B",
        "%Y %b",
        "%d %B %Y", 
        "%d %b %Y", 
        "%B %d, %Y",
        "%b %d, %Y" 
    ]
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            # Always return only year and month
            return datetime(parsed_date.year, parsed_date.month, 1)
        except ValueError:
            continue
    return None

# function to convert the years and months into a readable string.
def format_duration(duration):
    years = duration['years']
    months = duration['months']
    if years == 0:
        return f"{months} months"
    elif months == 0:
        return f"{years} years"
    else:
        return f"{years} years, {months} months"

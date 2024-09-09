import os
import re
import docx2txt
import tempfile
import json
import shutil
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Set up API keys 
google_api = os.environ["GOOGLE_API_KEY"]

class ResumeShortlistSystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=750,
                                                            separators=["\n\n", "\n# ", "\n- ", "\n\t"],
                                                            length_function=len)
    
    # Function to clear previous data
    def clear_previous_data():
        if os.path.exists("job_store"):
            shutil.rmtree("job_store")
        if os.path.exists("resume_stores"):
            shutil.rmtree("resume_stores")
        # if 'job_id' in st.session_state:
        #     del st.session_state['job_id']
        # if 'resume_ids' in st.session_state:
        #     del st.session_state['resume_ids']

    '''
    # Function for load the resume into vector DB FastAPI
    def load_document(self, file_path, store_name):
        text = ""

        if file_path.endswith('.pdf'):
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    text += page.page_content
                print(len(text))
                # extract text from Image
                if not text.strip():
                    loader = PyPDFLoader(file_path, extract_images=True)
                    pages = loader.load()
                    for page in pages:
                        text += page.page_content

            except Exception as e:
                print(f"Error extracting PDF text: {str(e)}")
        
        elif file_path.endswith('.docx'):
            try:
                text = docx2txt.process(file_path)
                
            except Exception as e:
                print(f"Error extracting DOCX text: {str(e)}")
        
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
            except Exception as e:
                print(f"Error reading text file: {str(e)}")

        chunks = self.text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        vector_store.save_local(store_name)

        return vector_store
    '''

    # '''
    # Function for load the resume into vector DB
    def load_document(self, file, store_name):
        text = ""
        print(file)
        if file.filename.endswith('.pdf'):
            print(file)
            try:
                print("hello")
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name
                
                print("hello2")

                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                for page in pages:
                    text += page.page_content

                print(text)

                # extract text from Image
                if not text.strip():
                    loader = PyPDFLoader(temp_file_path, extract_images=True)
                    pages = loader.load()
                    for page in pages:
                        text += page.page_content
                
                os.remove(temp_file_path)

            except Exception as e:
                print("hello3")
                print(f"Error extracting PDF text: {str(e)}")
        
        elif file.filename.endswith('.docx'):
            try:
                self.text = docx2txt.process(file)
                
            except Exception as e:
                print(f"Error extracting DOCX text: {str(e)}")
        
        else:
            try:
                self.text = file.read().decode('utf-8')
                
            except Exception as e:
                print(f"Error reading text file: {str(e)}")
        
        chunks = self.text_splitter.split_text(text)
        print(text)
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        print(text)
        vector_store.save_local(store_name)

        return vector_store
    # '''

    # function for Resume Matching and Result Generation
    def match_resumes(self, job_description_store, resume_store, resume_id, jd_id):
        
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

        job_retriever = job_description_store.as_retriever(search_kwargs={"k": 5})
        resume_retriever = resume_store.as_retriever(search_kwargs={"k": 5})

        current_date = datetime.now().strftime("%B %Y")
        
        job_docs = job_retriever.get_relevant_documents("job description")
        resume_docs = resume_retriever.get_relevant_documents("resume")
        
        job_description = " ".join([doc.page_content for doc in job_docs])
        resume = " ".join([doc.page_content for doc in resume_docs])
        
        match_analysis = self.llm.invoke(custom_prompt_template.format(
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
    def calculate_duration(self, start_date, end_date):
        if start_date == "No Experience" or end_date == "No Experience":
            return {"years": 0, "months": 0}
        
        if end_date.lower() == 'present':
            end_date = datetime.now()
        else:
            end_date = self.parse_date(end_date)
        
        start_date = self.parse_date(start_date)
        
        if start_date and end_date:
            duration = relativedelta(end_date, start_date)
            return {
                "years": duration.years,
                "months": duration.months
            }
        return {"years": 0, "months": 0}

    # function for Attempts to parse various date formats.
    def parse_date(self, date_str):
        date_formats = [
            "%Y-%m",
            "%Y-%m-%d",
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
    def format_duration(self, duration):
        years = duration['years']
        months = duration['months']
        if years == 0:
            return f"{months} months"
        elif months == 0:
            return f"{years} years"
        else:
            return f"{years} years, {months} months"
        
    # Function for covert % value into floating
    def percentage_to_float(self, percentage_str):
        try:
            return float(percentage_str.strip('%')) 
        except (ValueError, AttributeError):
            return 0.0  # Return 0.0 if conversion fails
    
    # Function for assigning the weights for job positions
    def assign_weights(self, job_position, custom_weights=None):
        if custom_weights:
            return custom_weights
        
        # Senior Level Roles
        if any(keyword in job_position.lower() for keyword in ['senior', 'lead']):
            return {'skills': 0.3, 'work_experience': 0.35, 'education': 0.1, 
                    'projects': 0.15, 'certifications': 0.1}
        # Engineer Level
        elif 'engineer' in job_position.lower():
            return {'skills': 0.35, 'work_experience': 0.25, 'education': 0.15, 
                    'projects': 0.15, 'certifications': 0.1}
        # Associate Level
        elif 'associate' in job_position.lower():
            return {'skills': 0.3, 'work_experience': 0.2, 'education': 0.3, 
                    'projects': 0.15, 'certifications': 0.05}
        # Intern Level
        elif 'intern' in job_position.lower():
            return {'skills': 0.3, 'work_experience': 0.1, 'education': 0.4, 
                    'projects': 0.2, 'certifications': 0.0}
        # Project Management
        elif (all(word in job_position.lower() for word in ['senior', 'project', 'manager']) or all(word in job_position.lower() for word in ['senior', 'product', 'manager'])):
            return {'skills': 0.35, 'work_experience': 0.35, 'education': 0.1, 
                    'projects': 0.1, 'certifications': 0.1}
        
        elif (all(word in job_position.lower() for word in ['associate', 'project', 'manager']) or all(word in job_position.lower() for word in ['associate', 'product', 'manager'])):
            return {'skills': 0.3, 'work_experience': 0.3, 'education': 0.1, 
                    'projects': 0.2, 'certifications': 0.1}
        
        elif (all(word in job_position.lower() for word in ['intern', 'project', 'manager']) or all(word in job_position.lower() for word in ['intern', 'product', 'manager'])):
            return {'skills': 0.25, 'work_experience': 0.1, 'education': 0.35, 
                    'projects': 0.25, 'certifications': 0.05}
        # Default or Unrecognized Position
        else:
            return {'skills': 0.25, 'work_experience': 0.25, 'education': 0.25, 
                    'projects': 0.25, 'certifications': 0.0}

    # Function for calculate weightage score for every CV
    def calculate_cv_score(self, row, custom_weights=None):

        job_position = row['job_position']
        weights = self.assign_weights(job_position, custom_weights)

        score = (
            weights['skills'] * row['skills_match'] +
            weights['work_experience'] * row['experience_match'] +
            weights['education'] * row['education_match'] +
            weights['projects'] * row['projects_match'] +
            weights['certifications'] * row['certifications_percentage']
        )
        
        return score
    
    # helper function for calculate total duration
    def get_total_duration(self, result):
        if result:
            # Handle the case where there's no work experience
            if isinstance(result['work_experience_matching']['relevant_job_roles'], str):
                # Assume it's "No Experience" or similar
                result['work_experience_matching']['total_duration'] = {"years": 0, "months": 0}
            else:
                # Recalculate durations
                for job in result['work_experience_matching']['relevant_job_roles']:
                    job['duration'] = self.calculate_duration(job['start_date'], job['end_date'])
                                    
                # Recalculate total duration
                total_months = sum(job['duration']['years'] * 12 + job['duration']['months'] 
                    for job in result['work_experience_matching']['relevant_job_roles'])
                result['work_experience_matching']['total_duration'] = {
                    "years": total_months // 12,
                    "months": total_months % 12
                    }
        
        return result

    # helper function to ensure all values are lists
    def ensure_list(self, value):
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [value]
        elif pd.isna(value) or value is None:
            return []
        else:
            return [str(value)]

    # function remove special characters
    def sanitize_filename(self, filename):
        # Remove all special characters, keep only English alphabetic letters and numbers
        return re.sub(r'[^a-zA-Z0-9]', '', filename)
    
    # function for create dataframe
    def final_df(self, results, custom_weights=None):
        
        # create dataframe
        df = pd.DataFrame([{
            'job_id': r['job_id'],
            'job_position': r['job_position'],
            'resume_id': r['resume_id'],
            'name': r['personal_info']['name'],
            'email': r['personal_info']['email'],
            'contact_number': self.ensure_list(r['personal_info']['contact_number']),
                        
            'technical_skills': self.ensure_list(r['skills_matching']['technical_skills']),
            'soft_skills': self.ensure_list(r['skills_matching']['soft_skills']),
            'skills_match': r['skills_matching']['skills_percentage'],
            'skills_analysis': r['skills_matching']['skills_analysis'],

            'relevant_job_roles': self.ensure_list(r['work_experience_matching']['job_roles']),
            'total_duration': self.format_duration(r['work_experience_matching']['total_duration']),
            'key_responsibilities': self.ensure_list(r['work_experience_matching']['key_responsibilities']),
            'experience_match': r['work_experience_matching']['experience_percentage'],
            'experience_analysis': r['work_experience_matching']['experience_analysis'],

            'relevant_projects': self.ensure_list(r['projects_matching']['relevant_projects']),
            'technologies_and_outcomes': r['projects_matching']['technologies_and_outcomes'],
            'projects_match': r['projects_matching']['projects_percentage'],
            'projects_analysis': r['projects_matching']['projects_analysis'],

            'education': self.ensure_list(r['education_matching']['education']),
            'achievements': self.ensure_list(r['education_matching']['achievements']),
            'extra_activities': self.ensure_list(r['education_matching']['extra_activities']),
            'education_match': r['education_matching']['education_percentage'],
            'education_analysis': r['education_matching']['education_analysis'],

            'certifications': self.ensure_list(r['professional_certifications_matching']['certifications']),
            'certifications_percentage': r['professional_certifications_matching']['certifications_percentage'],
            'certifications_analysis': r['professional_certifications_matching']['certifications_analysis'],

            'overall_match': r['final_matching']['final_overall_percentage'],
            'overall_analysis': r['final_matching']['final_analysis']
            } for r in results
        ])

        # Replace None with NaN for better DataFrame handling
        df = df.replace({None: pd.NA})
                
        # Convert list columns to string representations
        list_columns = ['technical_skills', 'soft_skills', 'relevant_job_roles', 
                        'key_responsibilities', 'relevant_projects', 'education', 
                        'achievements', 'extra_activities', 'certifications']

        for col in list_columns:
            df[col] = df[col].apply(lambda x: ', '.join(x) if x else 'Not Mentioned')
                
        columns_to_convert = ['skills_match', 'experience_match', 'education_match',
                              'projects_match', 'certifications_percentage', 'overall_match']

        for column in columns_to_convert:
            if column in df.columns:
                df[column] = df[column].apply(self.percentage_to_float)
        
        df['weighted_cv_score'] = df.apply(self.calculate_cv_score, axis=1, 
                                           custom_weights = custom_weights).round(2)

        return df
    
    # Function to validate weights
    def validate_weights(self, weights):
        total_weight = sum(weights.values())
        if total_weight != 1.0:
            return False, f"Total weight must equal 1.0. Current total: {total_weight}"
        return True, ""
    

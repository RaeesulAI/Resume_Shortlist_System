# import neccessary libraries 
import streamlit as st
import os
import re
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from DocumentManager import extract_text, pdf_setup
from EmbeddingManager import split_document, clear_previous_data, create_vector_store
from RetreivalAgent import match_resumes, calculate_duration, format_duration

# load the all environment variables from .env files
load_dotenv()

# configure API keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# helper function to ensure all values are lists
def ensure_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return [value]
    elif pd.isna(value) or value is None:
        return []
    else:
        return [str(value)]
    
# function remove special characters
def sanitize_filename(filename):
    # Remove all special characters, keep only English alphabetic letters and numbers
    return re.sub(r'[^a-zA-Z0-9]', '', filename)

# Streamlit app
def main():

    st.set_page_config(page_title = "Resume Expert")
    st.title("Resume Shortlisting System")

    job_description_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])
    resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], 
                                    accept_multiple_files=True)

    if st.button("Process and Match"):
        if job_description_file is not None and resume_files:
            with st.spinner("Processing..."):

                # Clear previous data
                clear_previous_data()

                # Generate unique ID for job description
                job_id = os.path.splitext(job_description_file.name)[0]

                # Process Job Description
                job_text = extract_text(job_description_file)
                if job_text is None:
                    st.error(f"Failed to extract text from job description file: {job_description_file.name}")
                    return
                job_chunks = split_document(job_text)
                job_store = create_vector_store(job_chunks, "job_store")
                print("\nSuccessfully completed the JD...\n")

                # Process Resumes
                results = []
                for resume_file in resume_files:
                    resume_id = os.path.splitext(resume_file.name)[0]
                    resume_id = sanitize_filename(resume_id) # remove special characters
                    resume_text = extract_text(resume_file)
                    if resume_text is None:
                        st.error(f"Failed to extract text from resume file: {resume_file.name}")
                        continue
                    else:
                        resume_chunks = split_document(resume_text)
                        # print(f"Number of resume chunks: {len(resume_chunks)}")
                        resume_store = create_vector_store(resume_chunks, f"resume_stores/{resume_id}")

                        # Match Resume with Job Description
                        match_result = match_resumes(job_store, resume_store, resume_id, job_id)
                        if match_result:
                            # Handle the case where there's no work experience
                            if isinstance(match_result['work_experience_matching']['relevant_job_roles'], str):
                                # Assume it's "No Experience" or similar
                                match_result['work_experience_matching']['total_duration'] = {"years": 0, "months": 0}
                            else:
                                # Recalculate durations
                                for job in match_result['work_experience_matching']['relevant_job_roles']:
                                    job['duration'] = calculate_duration(job['start_date'], job['end_date'])
                                
                                # Recalculate total duration
                                total_months = sum(job['duration']['years'] * 12 + job['duration']['months'] 
                                                for job in match_result['work_experience_matching']['relevant_job_roles'])
                                match_result['work_experience_matching']['total_duration'] = {
                                    "years": total_months // 12,
                                    "months": total_months % 12
                                }
                                
                            results.append(match_result)
                    
                    print("Successfully completed: " + resume_id + "\n")

                # Create DataFrame
                df = pd.DataFrame([
                    {
                        'job_id': r['job_id'],
                        'job_position': r['job_position'],
                        'resume_id': r['resume_id'],
                        'name': r['personal_info']['name'],
                        'email': r['personal_info']['email'],
                        'contact_number': ensure_list(r['personal_info']['contact_number']),
                        
                        'technical_skills': ensure_list(r['skills_matching']['technical_skills']),
                        'soft_skills': ensure_list(r['skills_matching']['soft_skills']),
                        'skills_match': r['skills_matching']['skills_percentage'],
                        'skills_analysis': r['skills_matching']['skills_analysis'],

                        'relevant_job_roles': ensure_list(r['work_experience_matching']['job_roles']),
                        'total_duration': format_duration(r['work_experience_matching']['total_duration']),
                        'key_responsibilities': ensure_list(r['work_experience_matching']['key_responsibilities']),
                        'experience_match': r['work_experience_matching']['experience_percentage'],
                        'experience_analysis': r['work_experience_matching']['experience_analysis'],

                        'relevant_projects': ensure_list(r['projects_matching']['relevant_projects']),
                        'technologies_and_outcomes': r['projects_matching']['technologies_and_outcomes'],
                        'projects_match': r['projects_matching']['projects_percentage'],
                        'projects_analysis': r['projects_matching']['projects_analysis'],

                        'education': ensure_list(r['education_matching']['education']),
                        'achievements': ensure_list(r['education_matching']['achievements']),
                        'extra_activities': ensure_list(r['education_matching']['extra_activities']),
                        'education_match': r['education_matching']['education_percentage'],
                        'education_analysis': r['education_matching']['education_analysis'],

                        'certifications': ensure_list(r['professional_certifications_matching']['certifications']),
                        'certifications_percentage': r['professional_certifications_matching']['certifications_percentage'],
                        'certifications_analysis': r['professional_certifications_matching']['certifications_analysis'],

                        'overall_match': r['final_matching']['final_overall_percentage'],
                        'overall_analysis': r['final_matching']['final_analysis']
                    } for r in results
                ])

                # Replace None with NaN for better DataFrame handling
                df = df.replace({None: pd.NA})
                
                # Convert list columns to string representations
                list_columns = ['technical_skills', 'soft_skills', 'relevant_job_roles', 'key_responsibilities', 
                                'relevant_projects', 'education', 'achievements', 'extra_activities', 
                                'certifications']

                for col in list_columns:
                    df[col] = df[col].apply(lambda x: ', '.join(x) if x else 'Not Mentioned')
                
                # Display Results
                st.subheader(f"Job Description: {job_id}")
                for result in results:
                    st.subheader(f"Match Analysis for Resume ID: {result['resume_id']}")
                    st.json(result)

                # Display DataFrame
                st.subheader("Matching Percentages Summary")
                st.dataframe(df)

                # Option to download DataFrame as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="resume_matching_results.csv",
                    mime="text/csv",
                )

        else:
            st.error("Please upload both a job description and at least one resume.")


if __name__ == "__main__":
    main()
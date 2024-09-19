# import neccessary libraries 
import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from testing import ResumeShortlistSystem

# load the all environment variables from .env files
load_dotenv()

# configure API keys
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the Knowledge-Based Search Retrieval System
RS_system = ResumeShortlistSystem()

# Streamlit app
def main():

    st.set_page_config(page_title = "Resume Expert")
    st.title("Resume Shortlisting System")

    job_description_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])
    resume_files = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], 
                                    accept_multiple_files=True)
    
    weightage = st.selectbox(
        "How woul you like to give weightage?",
        ("Default", "Custom"),
        index=None,
        placeholder="Select Weightage Method.."
    )

    custom_weightage = None

    if weightage == "Custom":
        
        st.write("Enter the Weightage value between 0 to 1")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            skills = st.number_input(
                "Skills", value=None, placeholder="Type between 0 to 1...", 
                max_value=1.0, min_value=0.0, format="%.2f"
                )
        
        with col2:
            experience = st.number_input(
                "Experience", value=None, placeholder="Type between 0 to 1...", 
                max_value=1.0, min_value=0.0, format="%.2f"
                )
        with col3:
            projects = st.number_input(
                "Projects", value=None, placeholder="Type between 0 to 1...", 
                max_value=1.0, min_value=0.0, format="%.2f"
                )
        with col4:
            education = st.number_input(
                "Education", value=None, placeholder="Type between 0 to 1...", 
                max_value=1.0, min_value=0.0, format="%.2f"
                )
        with col5:
            certification = st.number_input(
                "Certfication", value=None, placeholder="Type between 0 to 1...", 
                max_value=1.0, min_value=0.0, format="%.2f"
                )
    
        custom_weightage = {
            "skills": skills, 
            "work_experience": experience, 
            "projects": projects, 
            "education": education, 
            "certifications": certification
            }
        
        # Validate custom weights
        is_valid, error_message = RS_system.validate_weights(custom_weightage)
        if not is_valid:
            st.error(error_message)
            return

    if st.button("Process and Match"):
        if job_description_file is not None and resume_files:
            with st.spinner("Processing..."):

                # Clear previous data
                # RS_system.clear_previous_data

                # Generate unique ID for job description
                job_id = os.path.splitext(job_description_file.name)[0]

                # Process Job Description
                job_store = RS_system.load_document(job_description_file, f"job_store/{job_id}")
                print("Successfully completed the JD...\n")

                # Process Resumes
                results = []
                for resume_file in resume_files:
                    resume_id = os.path.splitext(resume_file.name)[0]
                    resume_id = RS_system.sanitize_filename(resume_id) # remove special characters
                    print(resume_id, " Process Started.......")
                    resume_store = RS_system.load_document(resume_file, f"resume_stores/{resume_id}")
                    match_result = RS_system.match_resumes(job_store, resume_store, resume_id, job_id)
                    match_result = RS_system.get_total_duration(match_result)
                    results.append(match_result)
                    
                    print("Successfully completed: " + resume_id + "\n")

                df = RS_system.final_df(results, custom_weights=custom_weightage)

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

                print("Successfully Completed the Process..")

        else:
            st.error("Please upload both a job description and at least one resume.")


if __name__ == "__main__":
    main()
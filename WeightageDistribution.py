import pandas as pd

def percentage_to_float(percentage_str):
    try:
        return float(percentage_str.strip('%')) 
    except (ValueError, AttributeError):
        return 0.0  # Return 0.0 if conversion fails

def assign_weights(job_position):
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

def calculate_cv_score(row):
    job_position = row['job_position']
    weights = assign_weights(job_position)

    score = (
        weights['skills'] * row['skills_match'] +
        weights['work_experience'] * row['experience_match'] +
        weights['education'] * row['education_match'] +
        weights['projects'] * row['projects_match'] +
        weights['certifications'] * row['certifications_percentage']
    )
    
    return score
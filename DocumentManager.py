import numpy as np
import docx2txt
import fitz
import pytesseract
import easyocr
import pdf2image
import base64
import io
from PIL import Image
from PyPDF2 import PdfReader
 

# function for convert file to text
def extract_text(file):
    text = ""
    if file.name.endswith('.pdf'):
        try:
            # Try PyPDF2 first
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text
                            
            # If PyPDF2 fails to extract text, try PyMuPDF
            if not text.strip():
                print("PyPDF2 failed to extract text. Trying PyMuPDF...")
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf_document:
                    page_text = page.get_text()
                    text += page_text
                    
                pdf_document.close()
            
            # If still no text, try OCR
            if not text.strip():
                print("Attempting OCR...")
                pdf_document = fitz.open(stream=file.read(), filetype="pdf")
                for page in pdf_document:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                    text += page_text
                    
                pdf_document.close()

        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
    elif file.name.endswith('.docx'):
        try:
            text = docx2txt.process(file)
        except Exception as e:
            print(f"Error extracting DOCX text: {str(e)}")
    else:
        try:
            text = file.read().decode('utf-8')
        except Exception as e:
            print(f"Error reading text file: {str(e)}")

    if len(text) == 0:
        print(f"Warning: No text could be extracted from {file.name}")
        return None
    else:
        print(f"Extracted {len(text)} characters from {file.name}")
    
    return text

# function for extract text using ICR
def extract_text_with_icr(file):
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])  # Use English language model
    # Open the PDF
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    
    # Iterate through each page
    for page in doc:
        # Convert the page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        # Perform OCR on the image
        result = reader.readtext(img_np)
        # Extract the text
        page_text = " ".join([text[1] for text in result])
        full_text += page_text
    
    return full_text

# function for setup the pdf file
def input_pdf_setup(file):

    if file is not None:
        # Convert the PDF to image
        images = pdf2image.convert_from_bytes(file.read())
        # 1st page of image
        first_page = images[0]
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        # first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    
    else:
        raise FileNotFoundError("No file uploaded")

# function for data ingestion
def ingest_data(job_description_file, resume_files):
    
    job_description = extract_text(job_description_file)
    resumes = [extract_text(resume_file) for resume_file in resume_files]
    
    return job_description, resumes
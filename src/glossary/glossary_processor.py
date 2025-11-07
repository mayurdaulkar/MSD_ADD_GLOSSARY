# Removed the validation part and still the code workign correctly

# import base64
# import json
# import re
# import time
# import shutil
# import os
# from pathlib import Path
# from langchain_openai import ChatOpenAI
# from pdf2image import convert_from_path
# from PIL import ImageEnhance, ImageFilter
# import psycopg2
# from typing import List, Optional
# import requests
# from .models import GlossaryEntry, GlossaryExtraction, ExtractRequest
# from .prompt import GLOSSARY_WITH_LANGUAGES, GLOSSARY_WITHOUT_LANGUAGES, RAW_TEXT_EXTRACTION, LANGUAGE_DETECTION



# # Import configuration
# from .config import (
#     OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE,
#     DB_CONFIG, PDF_DPI, PDF_THREAD_COUNT, PDF_OUTPUT_FOLDER, PDF_FORMAT,
#     IMAGE_CONTRAST, IMAGE_SHARPNESS, IMAGE_BRIGHTNESS, IMAGE_MEDIAN_FILTER_SIZE,
#     DEBUG, PORT, HOST, OUTPUT_JSON_FILE
# )




# # ==============================================================================
# # PHASE 1: PDF & IMAGE PROCESSING
# # ==============================================================================

# def preprocess_image(image):
#     """Preprocess an image for better OCR accuracy."""
#     # Convert to grayscale
#     image = image.convert('L')
    
#     # Enhance contrast
#     enhancer = ImageEnhance.Contrast(image)
#     image = enhancer.enhance(IMAGE_CONTRAST)
    
#     # Enhance sharpness
#     enhancer = ImageEnhance.Sharpness(image)
#     image = enhancer.enhance(IMAGE_SHARPNESS)
    
#     # Apply median filter
#     image = image.filter(ImageFilter.MedianFilter(size=IMAGE_MEDIAN_FILTER_SIZE))
    
#     # Enhance brightness
#     enhancer = ImageEnhance.Brightness(image)
#     image = enhancer.enhance(IMAGE_BRIGHTNESS)
    
#     return image


# def download_pdf_from_url(pdf_url: str, temp_folder: str) -> str:                                         # downloading the link of the pdf with this code
#     """Download PDF from URL and save to temporary folder."""
#     try:
#         Path(temp_folder).mkdir(parents=True, exist_ok=True)                          #data will be saved temporary based
        
#         response = requests.get(pdf_url, timeout=30)
#         response.raise_for_status()
        
#         # Validate PDF header
#         if response.content[:4] != b'%PDF':
#             raise ValueError("Downloaded file is not a valid PDF")
        
#         # Save downloaded PDF
#         filename = "downloaded_glossary.pdf"
#         filepath = Path(temp_folder) / filename
        
#         with open(filepath, 'wb') as f:
#             f.write(response.content)
        
#         print(f"✓ PDF downloaded successfully from URL\n")
#         return str(filepath)
        
#     except requests.exceptions.RequestException as e:
#         raise Exception(f"Failed to download PDF from URL: {e}")
#     except Exception as e:
#         raise Exception(f"Error processing downloaded PDF: {e}")


# def extract_images_from_pdf(pdf_path: str, output_folder: str, dpi: int = PDF_DPI, first_page=None, last_page=None) -> list:
#     """Convert PDF pages to images and save them."""
#     try:
#         Path(output_folder).mkdir(parents=True, exist_ok=True)
        
#         kwargs = {
#             'dpi': dpi,
#             'fmt': PDF_FORMAT,
#             'thread_count': PDF_THREAD_COUNT
#         }
        
#         if first_page:
#             kwargs['first_page'] = first_page
#         if last_page:
#             kwargs['last_page'] = last_page
        
#         print(f"  Converting PDF to images at {dpi} DPI...")
#         images = convert_from_path(pdf_path, **kwargs)
        
#         extracted_paths = []
#         for page_num, image in enumerate(images, first_page or 1):
#             image = preprocess_image(image)
#             filename = f"page_{page_num:03d}.jpg"
#             filepath = Path(output_folder) / filename
#             image.save(str(filepath), "jpeg", optimize=True)
#             extracted_paths.append(str(filepath))
        
#         print(f"  ✓ Extracted {len(images)} images from PDF\n")
#         return extracted_paths
        
#     except ImportError as e:
#         print(f"  ✗ Missing dependency: {e}")
#         print("    Install: pip install pdf2image pillow")
#         return []
#     except Exception as e:
#         print(f"  ✗ Error extracting images from PDF: {e}")
#         return []


# # ==============================================================================
# # PHASE 2: EXTRACT TEXT FROM IMAGE WITH STRUCTURED OUTPUT
# # ==============================================================================

# def extract_text_from_image(image_path: str, llm: ChatOpenAI, source_lang: str = None, target_langs: list = None) -> GlossaryExtraction:
#     """Extract structured glossary data from an image using with_structured_output."""
#     try:
#         image_data = Path(image_path).read_bytes()
#         base64_image = base64.standard_b64encode(image_data).decode("utf-8")
        
#         extension = Path(image_path).suffix.lower()
#         media_type_map = {
#             '.jpg': 'image/jpeg',
#             '.jpeg': 'image/jpeg',
#             '.png': 'image/png',
#             '.gif': 'image/gif',
#             '.webp': 'image/webp'
#         }
#         media_type = media_type_map.get(extension, 'image/jpeg')
        

#        # Build prompt based on whether languages are provided
#         if source_lang and target_langs:
#             prompt_text = GLOSSARY_WITH_LANGUAGES.format(
#                 source_lang=source_lang,
#                 target_langs=', '.join(target_langs)
#             )
#         else:
#             prompt_text = GLOSSARY_WITHOUT_LANGUAGES
#         # Use structured output
#         structured_chain = llm.with_structured_output(GlossaryExtraction)                                                            #have used the with structured output
#         result = structured_chain.invoke([
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}},
#                     {"type": "text", "text": prompt_text}
#                 ],
#             }
#         ])
        
#         return result
        
#     except Exception as e:
#         print(f"    ✗ Error extracting from image: {e}")
#         return GlossaryExtraction(rows=[])


# # ==============================================================================
# # PHASE 3: LANGUAGE DETECTION & VALIDATION (NO PREDEFINED LANGUAGE RESTRICTIONS)
# # ==============================================================================

# def detect_languages_from_raw_image(image_path: str, llm: ChatOpenAI) -> list:
#     """Extract raw text from image for language detection."""
#     try:
#         image_data = Path(image_path).read_bytes()
#         base64_image = base64.standard_b64encode(image_data).decode("utf-8")
        
#         extension = Path(image_path).suffix.lower()
#         media_type_map = {
#             '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
#             '.gif': 'image/gif', '.webp': 'image/webp'
#         }
#         media_type = media_type_map.get(extension, 'image/jpeg')
        
#         response = llm.invoke([
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}},
#                     {"type": "text", "text": "" + RAW_TEXT_EXTRACTION}
#                 ],
#             }
#         ])
        
#         raw_text = response.content if hasattr(response, 'content') else str(response)
#         return detect_languages(raw_text, llm)                                                            
        
#     except Exception as e:
#         print(f"    ✗ Error in raw text extraction: {e}")
#         return []


# def detect_languages(raw_text: str, llm: ChatOpenAI) -> list:
#     """Detect languages from raw text using LLM - NO PREDEFINED LANGUAGE RESTRICTIONS."""
#     if not raw_text or len(raw_text.strip()) < 50:
#         print("    [Detection] Insufficient text for language detection")
#         return []
    
#     print("    [Detection] Analyzing languages in extracted text...")
    
#     try:
#         response = llm.invoke([
#             {
#                 "role": "user",
#                 "content": LANGUAGE_DETECTION.format(raw_text=raw_text[:3000])
#             }
#         ])
        
#         response_text = response.content if hasattr(response, 'content') else str(response)
        
#         json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(0)
#             languages = json.loads(json_str)
            
#             # Accept ANY detected language without restriction
#             detected_languages = []
#             for lang in languages:
#                 if isinstance(lang, str):
#                     lang_clean = lang.lower().strip()
#                     if lang_clean and lang_clean not in detected_languages:
#                         detected_languages.append(lang_clean)
            
#             if detected_languages:
#                 print(f"    [Detection] ✓ Languages found: {detected_languages}")
#                 return detected_languages
        
#         print("    [Detection] ✗ No languages detected in response")
#         return []
        
#     except Exception as e:
#         print(f"    [Detection] ✗ Error: {e}")
#         return []


# def validate_languages(source_lang: str, target_langs: list, detected_langs: list) -> bool:
#     """Validate if provided languages match detected languages (no predefined restrictions)."""
#     source_norm = source_lang.lower().strip()
#     target_norm = [t.lower().strip() for t in target_langs]
#     detected_norm = [d.lower().strip() for d in detected_langs]
    
#     print("\n  [Validation] Checking language match...")
#     print(f"    Provided - Source: {source_norm}, Targets: {target_norm}")
#     print(f"    Detected in PDF: {detected_norm}")
    
#     # Check if source language is in detected languages
#     if source_norm not in detected_norm:
#         print(f"  ✗ ERROR: Source language '{source_lang}' NOT found in PDF")
#         print(f"    Detected languages: {detected_norm}")
#         return False
    
#     # Check if all target languages are in detected languages
#     for target in target_norm:
#         if target not in detected_norm:
#             print(f"  ✗ ERROR: Target language '{target}' NOT found in PDF")
#             print(f"    Detected languages: {detected_norm}")
#             return False
    
#     print(f"  ✓ Language validation passed!")
#     return True


# # ==============================================================================
# # PHASE 4: PARSING WITH PYDANTIC VALIDATION
# # ==============================================================================

# def parse_glossary_text(structured_result: GlossaryExtraction, source_language: str, target_languages: list) -> List[GlossaryEntry]:      #used to check if the correct format is there or not for the structured_output
#     """Return validated Pydantic entries from structured output."""
#     entries = []
    
#     try:
#         if not structured_result or not structured_result.rows:
#             return entries
        
#         for entry in structured_result.rows:
#             # Basic validation: ensure the core fields are not empty
#             if entry.source_text and entry.source_text.strip() and entry.translated_text and entry.translated_text.strip():
#                 entries.append(entry)
#     except Exception as e:
#         print(f"    ✗ Error processing entries: {e}")
    
#     return entries


# # ==============================================================================
# # PHASE 5: DATABASE FUNCTIONS
# # ==============================================================================

# def connect_to_database(db_config: dict):
#     """Connect to PostgreSQL database."""
#     try:
#         conn = psycopg2.connect(
#             host=db_config['host'],
#             port=db_config['port'],
#             database=db_config['name'],
#             user=db_config['user'],
#             password=db_config['password']
#         )
#         print("✓ Connected to database")
#         return conn
#     except Exception as e:
#         print(f"✗ Database connection failed: {e}")
#         return None


# def create_table_if_not_exists(connection, table_name: str) -> bool:
#     """Create glossary table if it doesn't exist."""
#     try:
#         cursor = connection.cursor()
        
#         create_sql = f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             id SERIAL PRIMARY KEY,
#             source_language VARCHAR(50),
#             target_language VARCHAR(50),
#             source_text TEXT,
#             translated_text TEXT,
#             legal_reference TEXT
#         );
#         """
        
#         cursor.execute(create_sql)
#         connection.commit()
#         print(f"✓ Table '{table_name}' ready")
#         cursor.close()
#         return True
        
#     except Exception as e:
#         print(f"✗ Table creation failed: {e}")
#         connection.rollback()
#         return False


# def insert_json_to_database(connection, table_name: str, json_data: list) -> bool:
#     """Insert JSON glossary entries into database."""
#     try:
#         cursor = connection.cursor()
        
#         insert_sql = f"""
#         INSERT INTO {table_name} 
#         (source_language, target_language, source_text, translated_text, legal_reference)
#         VALUES (%s, %s, %s, %s, %s)
#         """
        
#         records = []
#         for item in json_data:
#             if isinstance(item, GlossaryEntry):
#                 item = item.dict()
            
#             records.append((
#                 item.get('source_language', ''),
#                 item.get('target_language', ''),
#                 item.get('source_text', ''),
#                 item.get('translated_text', ''),
#                 item.get('legal_reference', '')
#             ))
        
#         cursor.executemany(insert_sql, records)
#         connection.commit()
        
#         cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
#         total = cursor.fetchone()[0]
        
#         print(f"✓ Records inserted successfully")
#         print(f"  Total entries in database: {total}")
        
#         cursor.close()
#         return True
        
#     except Exception as e:
#         print(f"✗ Insert failed: {e}")
#         connection.rollback()
#         return False


# def save_to_database(json_data: list, table_name: str) -> bool:
#     """Save glossary JSON data to database."""
#     conn = connect_to_database(DB_CONFIG)
#     if not conn:
#         return False
    
#     if not create_table_if_not_exists(conn, table_name):
#         conn.close()
#         return False
    
#     success = insert_json_to_database(conn, table_name, json_data)
#     conn.close()
    
#     return success


# # ==============================================================================
# # HELPER FUNCTIONS FOR RESPONSE FORMATTING
# # ==============================================================================

# def format_error_response(error_msg: str) -> dict:                          #they simply give the standard error response if any error occur like pdf not found and all
#     """Format error response"""
#     return {
#         "message": error_msg,
#         "status": "Error",
#         "data": [],
#         "table_name": None
#     }


# def format_success_response(data: list, entry_count: int, table_name: str = None) -> dict:                  #provide the success messsage
#     """Format success response"""
#     message = f"Glossary created successfully. Extracted {entry_count} entries."
#     if table_name:
#         message += f" Data stored in table: {table_name}"
    
#     return {
#         "message": message,
#         "status": "Success",
#         "data": data,
#         "table_name": table_name
#     }


# # ==============================================================================
# # VALIDATION FUNCTIONS
# # ==============================================================================

# def validate_pdf_file(file_path: str) -> tuple:
#     """Strictly validate that the file is a valid PDF."""
#     pdf_file = Path(file_path)
    
#     # Check if file exists
#     if not pdf_file.exists():
#         return False, f"File not found: {file_path}"
    
#     # Check if file extension is .pdf
#     if pdf_file.suffix.lower() != '.pdf':
#         return False, f"Invalid file type. Only PDF files are accepted. Got: {pdf_file.suffix}"
    
#     # Check if file is readable and is actually a PDF
#     try:
#         with open(file_path, 'rb') as f:
#             header = f.read(4)
#             if header != b'%PDF':
#                 return False, "File is not a valid PDF (invalid header)"
#     except Exception as e:
#         return False, f"Cannot read file: {e}"
    
#     return True, "Valid PDF file"


# def is_valid_url(url: str) -> bool:
#     """Check if the input is a valid URL."""
#     return url.startswith('http://') or url.startswith('https://')


# def validate_request_data(data):                              #use to provide the validation for the incoming request data
#     """Validate incoming request data."""
#     errors = []
    
#     # Check required fields
#     if not data.get('source_language') or not isinstance(data.get('source_language'), str):
#         errors.append("source_language is required and must be a string")
    
#     if not data.get('list_target_languages') or not isinstance(data.get('list_target_languages'), list):
#         errors.append("list_target_languages is required and must be a list")
    
#     if not data.get('pdf_path') or not isinstance(data.get('pdf_path'), str):
#         errors.append("pdf_path is required and must be a string (URL or local path)")
    
#     # Check optional fields
#     insert_db = data.get('insert_db', False)
#     if not isinstance(insert_db, bool):
#         if isinstance(insert_db, str):
#             insert_db = insert_db.lower() in ['true', '1', 'yes']
#         else:
#             errors.append("insert_db must be a boolean or string ('true'/'false')")
    
#     first_page = data.get('first_page')
#     last_page = data.get('last_page')
    
#     if first_page is not None and not isinstance(first_page, int):
#         errors.append("first_page must be an integer or null")
    
#     if last_page is not None and not isinstance(last_page, int):
#         errors.append("last_page must be an integer or null")
    
#     return errors, insert_db


# # ==============================================================================
# # MAIN GLOSSARY EXTRACTION FUNCTION
# # ==============================================================================

# def add_glossary(source_language: str, list_target_languages: list, pdf_path: str, insert_db: bool, first_page=None, last_page=None) -> dict:                       #add_glossary function 

#     """
#     Main glossary extraction function - PDF from URL or Local Path
    
#     Args:
#         source_language: Source language (e.g., "english")
#         list_target_languages: List of target languages (e.g., ["hindi", "gujarati"])
#         pdf_path: URL or local file path to PDF
#         insert_db: Boolean to insert into database or not
#         first_page: Optional first page number
#         last_page: Optional last page number
    
#     Returns:
#         Dict with format: {"message": "...", "status": "...", "data": [...], "table_name": "..."}
#     """
    
#     print("\n" + "="*60)
#     print("GLOSSARY EXTRACTION - PDF (URL or LOCAL)")
#     print("="*60)
    
#     # STEP 1: VALIDATE INPUTS
#     print("\n[STEP 1] Validating inputs...")
    
#     if not source_language or not list_target_languages:
#         return format_error_response("Source and target languages are required")
    
#     source_lang_lower = source_language.lower()
#     target_langs_lower = [lang.lower() for lang in list_target_languages]
    
#     if source_lang_lower in target_langs_lower:
#         return format_error_response("Source language cannot be in target languages")
    
#     print(f"✓ Source Language: {source_language}")
#     print(f"✓ Target Languages: {', '.join(list_target_languages)}")
    
#     # STEP 2: VALIDATE PDF FILE
#     print("\n[STEP 2] Validating PDF source (URL or LOCAL)...")
    
#     local_pdf_path = pdf_path
#     download_folder = None
    
#     # Check if it's a URL
#     if is_valid_url(pdf_path):
#         print(f"✓ PDF source detected as URL")
#         download_folder = Path(PDF_OUTPUT_FOLDER) / f"downloads_{int(time.time())}"
        
#         try:
#             local_pdf_path = download_pdf_from_url(pdf_path, str(download_folder))
#             print(f"✓ PDF downloaded successfully")
#         except Exception as e:
#             return format_error_response(f"Failed to download PDF: {str(e)}")
#     else:
#         # Local file validation
#         is_valid, validation_msg = validate_pdf_file(pdf_path)
        
#         if not is_valid:
#             return format_error_response(f"PDF validation failed: {validation_msg}")
        
#         pdf_file = Path(pdf_path)
#         print(f"✓ PDF validation passed: {pdf_file.name}")
    
#     # STEP 3: INITIALIZE LLM
#     print("\n[STEP 3] Initializing LLM...")
    
#     if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-DUMMYKEY":
#         if download_folder:
#             shutil.rmtree(download_folder, ignore_errors=True)
#         return format_error_response("OPENAI_API_KEY not configured. Set it in your .env file.")
    
#     llm = ChatOpenAI(
#         model=OPENAI_MODEL,
#         temperature=OPENAI_TEMPERATURE,
#         max_tokens=OPENAI_MAX_TOKENS,
#         api_key=OPENAI_API_KEY
#     )
#     print("✓ LLM initialized")
    
#     # STEP 4: EXTRACT IMAGES FROM PDF
#     print("\n[STEP 4] Converting PDF to images...")
    
#     temp_folder = Path(PDF_OUTPUT_FOLDER) / "temp_images"                      #where the glossary is stored
    
#     extracted_images = extract_images_from_pdf(
#         local_pdf_path,
#         str(temp_folder),
#         PDF_DPI,
#         first_page,
#         last_page
#     )
    
#     if not extracted_images:
#         if download_folder:
#             shutil.rmtree(download_folder, ignore_errors=True)
#         shutil.rmtree(temp_folder, ignore_errors=True)
#         return format_error_response("Failed to extract images from PDF")
    
#     print(f"✓ Extracted {len(extracted_images)} images")
    
#     # STEP 5: DETECT AND VALIDATE LANGUAGES (NO PREDEFINED RESTRICTIONS)
#     print("\n[STEP 5] Detecting and validating languages...")
    
#     detected_langs = detect_languages_from_raw_image(extracted_images[0], llm)
    
#     if not detected_langs:
#         if download_folder:
#             shutil.rmtree(download_folder, ignore_errors=True)
#         shutil.rmtree(temp_folder, ignore_errors=True)
#         return format_error_response("Could not detect any languages in PDF")
    
#     # VALIDATION: Check if provided languages match detected languages
#     is_valid_langs = validate_languages(source_language, list_target_languages, detected_langs)
    
#     if not is_valid_langs:
#         if download_folder:
#             shutil.rmtree(download_folder, ignore_errors=True)
#         shutil.rmtree(temp_folder, ignore_errors=True)
#         return format_error_response(
#             f"Language mismatch! Provided languages do not match PDF content. "
#             f"PDF contains: {detected_langs}"
#         )
    
#     # STEP 6: EXTRACT GLOSSARY FROM ALL IMAGES
#     print("\n[STEP 6] Extracting glossary entries...")
    
#     all_entries = []
    
#     for idx, img_path in enumerate(extracted_images, 1):
#         print(f"  [{idx}/{len(extracted_images)}] Processing {Path(img_path).name}...")
        
#         try:
#             extracted_glossary = extract_text_from_image(
#                 img_path,
#                 llm,
#                 source_language,
#                 list_target_languages
#             )
            
#             entries = parse_glossary_text(
#                 extracted_glossary,
#                 source_language,
#                 list_target_languages
#             )
            
#             all_entries.extend(entries)
#             print(f"    ✓ Extracted {len(entries)} entries")
        
#         except Exception as e:
#             print(f"    ✗ Error: {e}")
#             continue
    
#     if not all_entries:
#         if download_folder:
#             shutil.rmtree(download_folder, ignore_errors=True)
#         shutil.rmtree(temp_folder, ignore_errors=True)
#         return format_error_response("No glossary entries could be extracted from PDF")
    
#     print(f"✓ Total entries extracted: {len(all_entries)}")
    

#     table_name = None
    
#     print("\n[STEP 7] Database operation...")
    
#     if insert_db:
#         print("  Inserting into database...")
#         table_name = f"glossary_new_one"                                                                                #here is the name of the table of database
        
#         try:
#             json_data = [entry.dict() for entry in all_entries]
#             if save_to_database(json_data, table_name):
#                 print(f"  ✓ Database insertion successful in table: {table_name}")
#             else:
#                 print("  ✗ Database insertion failed (but extraction succeeded)")
#                 table_name = None
#         except Exception as e:
#             print(f"  ✗ Database error: {e}")
#             table_name = None
#     else:
#         print("  Skipping database insertion (insert_db=False)")
    
#     # STEP 8: CLEANUP
#     print("\n[STEP 8] Cleaning up temporary files...")
#     try:
#         # Delete temp_images folder (contains all extracted images)
#         if temp_folder and os.path.exists(str(temp_folder)):
#             shutil.rmtree(str(temp_folder), ignore_errors=False)
#             print("✓ Temporary images folder 'temp_images' deleted successfully")

#             # Also delete parent folder if empty
#             parent_dir = Path(temp_folder).parent
#             if parent_dir.exists() and not any(parent_dir.iterdir()):
#                 parent_dir.rmdir()
#                 print(f"✓ Parent folder '{parent_dir}' deleted (was empty)")

#         # Delete download_folder if it exists (for URL downloads)
#         if download_folder and os.path.exists(str(download_folder)):
#             shutil.rmtree(str(download_folder), ignore_errors=False)
#             print("✓ Downloaded PDF folder deleted successfully")

#     except Exception as e:
#         print(f"✗ Error while deleting temporary files: {e}")

    
#     # STEP 9: FORMAT RESPONSE
#     print("\n[STEP 9] Formatting response...")
    
#     output_data = [entry.dict() for entry in all_entries]
    
#     response = format_success_response(output_data, len(all_entries), table_name)
    
#     print("\n" + "="*60)
#     print(f"✓ EXTRACTION COMPLETE - {len(all_entries)} entries")
#     if table_name:
#         print(f"✓ Data stored in table: {table_name}")
#     print("="*60 + "\n")
    
#     return response


































\
# PARALLEL PROCESSING IMPLEMENTATION FOR FASTER EXTRACTION

import base64
import json
import re
import time
import shutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
from PIL import ImageEnhance, ImageFilter
import psycopg2
from typing import List, Optional
import requests
from .models import GlossaryEntry, GlossaryExtraction, ExtractRequest
from .prompt import GLOSSARY_WITH_LANGUAGES, GLOSSARY_WITHOUT_LANGUAGES, RAW_TEXT_EXTRACTION, LANGUAGE_DETECTION



# Import configuration
from .config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE,
    DB_CONFIG, PDF_DPI, PDF_THREAD_COUNT, PDF_OUTPUT_FOLDER, PDF_FORMAT,
    IMAGE_CONTRAST, IMAGE_SHARPNESS, IMAGE_BRIGHTNESS, IMAGE_MEDIAN_FILTER_SIZE,
    DEBUG, PORT, HOST, OUTPUT_JSON_FILE
)

# PARALLEL PROCESSING CONFIGURATION
MAX_WORKERS = 8  # Number of parallel threads (adjust based on your system)




# ==============================================================================
# PHASE 1: PDF & IMAGE PROCESSING
# ==============================================================================

def preprocess_image(image):
    """Preprocess an image for better OCR accuracy."""
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(IMAGE_CONTRAST)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(IMAGE_SHARPNESS)
    
    # Apply median filter
    image = image.filter(ImageFilter.MedianFilter(size=IMAGE_MEDIAN_FILTER_SIZE))
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(IMAGE_BRIGHTNESS)
    
    return image


def download_pdf_from_url(pdf_url: str, temp_folder: str) -> str:                                         
    """Download PDF from URL and save to temporary folder."""
    try:
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
        
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Validate PDF header
        if response.content[:4] != b'%PDF':
            raise ValueError("Downloaded file is not a valid PDF")
        
        # Save downloaded PDF
        filename = "downloaded_glossary.pdf"
        filepath = Path(temp_folder) / filename
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ PDF downloaded successfully from URL\n")
        return str(filepath)
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download PDF from URL: {e}")
    except Exception as e:
        raise Exception(f"Error processing downloaded PDF: {e}")


def extract_images_from_pdf(pdf_path: str, output_folder: str, dpi: int = PDF_DPI, first_page=None, last_page=None) -> list:
    """Convert PDF pages to images and save them."""
    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        kwargs = {
            'dpi': dpi,
            'fmt': PDF_FORMAT,
            'thread_count': PDF_THREAD_COUNT
        }
        
        if first_page:
            kwargs['first_page'] = first_page
        if last_page:
            kwargs['last_page'] = last_page
        
        print(f"  Converting PDF to images at {dpi} DPI...")
        images = convert_from_path(pdf_path, **kwargs)
        
        extracted_paths = []
        for page_num, image in enumerate(images, first_page or 1):
            image = preprocess_image(image)
            filename = f"page_{page_num:03d}.jpg"
            filepath = Path(output_folder) / filename
            image.save(str(filepath), "jpeg", optimize=True)
            extracted_paths.append(str(filepath))
        
        print(f"  ✓ Extracted {len(images)} images from PDF\n")
        return extracted_paths
        
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("    Install: pip install pdf2image pillow")
        return []
    except Exception as e:
        print(f"  ✗ Error extracting images from PDF: {e}")
        return []


# ==============================================================================
# PHASE 2: EXTRACT TEXT FROM IMAGE WITH STRUCTURED OUTPUT
# ==============================================================================

def extract_text_from_image(image_path: str, llm: ChatOpenAI, source_lang: str = None, target_langs: list = None) -> tuple:
    """Extract structured glossary data from an image using with_structured_output.
    
    Returns tuple: (image_path, GlossaryExtraction, entries_count)
    """
    try:
        image_data = Path(image_path).read_bytes()
        base64_image = base64.standard_b64encode(image_data).decode("utf-8")
        
        extension = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')
        

       # Build prompt based on whether languages are provided
        if source_lang and target_langs:
            prompt_text = GLOSSARY_WITH_LANGUAGES.format(
                source_lang=source_lang,
                target_langs=', '.join(target_langs)
            )
        else:
            prompt_text = GLOSSARY_WITHOUT_LANGUAGES
            
        # Use structured output
        structured_chain = llm.with_structured_output(GlossaryExtraction)
        result = structured_chain.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ])
        
        entries = parse_glossary_text(result, source_lang, target_langs)
        return (image_path, result, len(entries))
        
    except Exception as e:
        print(f"    ✗ Error extracting from image: {e}")
        return (image_path, GlossaryExtraction(rows=[]), 0)


# ==============================================================================
# PHASE 3: LANGUAGE DETECTION & VALIDATION (NO PREDEFINED LANGUAGE RESTRICTIONS)
# ==============================================================================

def detect_languages_from_raw_image(image_path: str, llm: ChatOpenAI) -> list:
    """Extract raw text from image for language detection."""
    try:
        image_data = Path(image_path).read_bytes()
        base64_image = base64.standard_b64encode(image_data).decode("utf-8")
        
        extension = Path(image_path).suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')
        
        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}},
                    {"type": "text", "text": "" + RAW_TEXT_EXTRACTION}
                ],
            }
        ])
        
        raw_text = response.content if hasattr(response, 'content') else str(response)
        return detect_languages(raw_text, llm)                                                            
        
    except Exception as e:
        print(f"    ✗ Error in raw text extraction: {e}")
        return []


def detect_languages(raw_text: str, llm: ChatOpenAI) -> list:
    """Detect languages from raw text using LLM - NO PREDEFINED LANGUAGE RESTRICTIONS."""
    if not raw_text or len(raw_text.strip()) < 50:
        print("    [Detection] Insufficient text for language detection")
        return []
    
    print("    [Detection] Analyzing languages in extracted text...")
    
    try:
        response = llm.invoke([
            {
                "role": "user",
                "content": LANGUAGE_DETECTION.format(raw_text=raw_text[:3000])
            }
        ])
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            languages = json.loads(json_str)
            
            # Accept ANY detected language without restriction
            detected_languages = []
            for lang in languages:
                if isinstance(lang, str):
                    lang_clean = lang.lower().strip()
                    if lang_clean and lang_clean not in detected_languages:
                        detected_languages.append(lang_clean)
            
            if detected_languages:
                print(f"    [Detection] ✓ Languages found: {detected_languages}")
                return detected_languages
        
        print("    [Detection] ✗ No languages detected in response")
        return []
        
    except Exception as e:
        print(f"    [Detection] ✗ Error: {e}")
        return []


def validate_languages(source_lang: str, target_langs: list, detected_langs: list) -> bool:
    """Validate if provided languages match detected languages (no predefined restrictions)."""
    source_norm = source_lang.lower().strip()
    target_norm = [t.lower().strip() for t in target_langs]
    detected_norm = [d.lower().strip() for d in detected_langs]
    
    print("\n  [Validation] Checking language match...")
    print(f"    Provided - Source: {source_norm}, Targets: {target_norm}")
    print(f"    Detected in PDF: {detected_norm}")
    
    # Check if source language is in detected languages
    if source_norm not in detected_norm:
        print(f"  ✗ ERROR: Source language '{source_lang}' NOT found in PDF")
        print(f"    Detected languages: {detected_norm}")
        return False
    
    # Check if all target languages are in detected languages
    for target in target_norm:
        if target not in detected_norm:
            print(f"  ✗ ERROR: Target language '{target}' NOT found in PDF")
            print(f"    Detected languages: {detected_norm}")
            return False
    
    print(f"  ✓ Language validation passed!")
    return True


# ==============================================================================
# PHASE 4: PARSING WITH PYDANTIC VALIDATION
# ==============================================================================

def parse_glossary_text(structured_result: GlossaryExtraction, source_language: str, target_languages: list) -> List[GlossaryEntry]:
    """Return validated Pydantic entries from structured output."""
    entries = []
    
    try:
        if not structured_result or not structured_result.rows:
            return entries
        
        for entry in structured_result.rows:
            # Basic validation: ensure the core fields are not empty
            if entry.source_text and entry.source_text.strip() and entry.translated_text and entry.translated_text.strip():
                entries.append(entry)
    except Exception as e:
        print(f"    ✗ Error processing entries: {e}")
    
    return entries


# ==============================================================================
# PHASE 5: PARALLEL EXTRACTION FUNCTION
# ==============================================================================
def extract_glossary_parallel(image_paths: list, llm: ChatOpenAI, source_language: str, target_languages: list, max_workers: int = MAX_WORKERS) -> List[GlossaryEntry]:
    """
    Extract glossary from multiple images in PARALLEL using ThreadPoolExecutor.
    BUT process results in SEQUENTIAL ORDER (1, 2, 3, 4...)
    
    How it works:
    1. Creates a thread pool with MAX_WORKERS threads
    2. Submits all image extraction tasks to the pool (workers start immediately)
    3. Each thread processes one image independently (in parallel)
    4. Results are collected in SEQUENTIAL ORDER (always 1→2→3→4...)
    5. Returns all entries combined
    
    Speedup: Sequential (10 images × 10s = 100s) vs Parallel (~10-15s) = ~7-10x faster!
    Output: Always printed in order 1, 2, 3... (not in completion order)
    """
    all_entries = []
    
    print(f"  Starting parallel extraction with {max_workers} workers...\n")
    
    # Create ThreadPoolExecutor context manager
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store futures in a LIST (maintains order)
        futures = [
            executor.submit(extract_text_from_image, img_path, llm, source_language, target_languages)
            for img_path in image_paths
        ]
        
        # Process results in SEQUENTIAL ORDER (image 1, then 2, then 3...)
        # This is the KEY DIFFERENCE - we iterate through futures in order, not as_completed
        for completed, future in enumerate(futures, start=1):
            img_path = image_paths[completed - 1]  # Get original image path (in order)
            
            try:
                # Get result from completed task
                # NOTE: This WAITS if the image hasn't finished processing yet
                image_path, extracted_glossary, entry_count = future.result()
                all_entries.extend(parse_glossary_text(extracted_glossary, source_language, target_languages))
                
                progress_percent = (completed / len(image_paths)) * 100
                print(f"  [{completed}/{len(image_paths)}] ({progress_percent:.0f}%) ✓ {Path(img_path).name} - {entry_count} entries extracted")
                
            except Exception as e:
                print(f"  [{completed}/{len(image_paths)}] ✗ Error processing {Path(img_path).name}: {e}")
                continue
    
    return all_entries


# ==============================================================================
# PHASE 6: DATABASE FUNCTIONS
# ==============================================================================

def connect_to_database(db_config: dict):
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['name'],
            user=db_config['user'],
            password=db_config['password']
        )
        print("✓ Connected to database")
        return conn
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None


def create_table_if_not_exists(connection, table_name: str) -> bool:
    """Create glossary table if it doesn't exist."""
    try:
        cursor = connection.cursor()
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            source_language VARCHAR(50),
            target_language VARCHAR(50),
            source_text TEXT,
            translated_text TEXT,
            legal_reference TEXT
        );
        """
        
        cursor.execute(create_sql)
        connection.commit()
        print(f"✓ Table '{table_name}' ready")
        cursor.close()
        return True
        
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        connection.rollback()
        return False


def insert_json_to_database(connection, table_name: str, json_data: list) -> bool:
    """Insert JSON glossary entries into database."""
    try:
        cursor = connection.cursor()
        
        insert_sql = f"""
        INSERT INTO {table_name} 
        (source_language, target_language, source_text, translated_text, legal_reference)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        records = []
        for item in json_data:
            if isinstance(item, GlossaryEntry):
                item = item.dict()
            
            records.append((
                item.get('source_language', ''),
                item.get('target_language', ''),
                item.get('source_text', ''),
                item.get('translated_text', ''),
                item.get('legal_reference', '')
            ))
        
        cursor.executemany(insert_sql, records)
        connection.commit()
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total = cursor.fetchone()[0]
        
        print(f"✓ Records inserted successfully")
        print(f"  Total entries in database: {total}")
        
        cursor.close()
        return True
        
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        connection.rollback()
        return False


def save_to_database(json_data: list, table_name: str) -> bool:
    """Save glossary JSON data to database."""
    conn = connect_to_database(DB_CONFIG)
    if not conn:
        return False
    
    if not create_table_if_not_exists(conn, table_name):
        conn.close()
        return False
    
    success = insert_json_to_database(conn, table_name, json_data)
    conn.close()
    
    return success


# ==============================================================================
# HELPER FUNCTIONS FOR RESPONSE FORMATTING
# ==============================================================================

def format_error_response(error_msg: str) -> dict:
    """Format error response"""
    return {
        "message": error_msg,
        "status": "Error",
        "data": [],
        "table_name": None
    }


def format_success_response(data: list, entry_count: int, table_name: str = None) -> dict:
    """Format success response"""
    message = f"Glossary created successfully. Extracted {entry_count} entries."
    if table_name:
        message += f" Data stored in table: {table_name}"
    
    return {
        "message": message,
        "status": "Success",
        "data": data,
        "table_name": table_name
    }


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_pdf_file(file_path: str) -> tuple:
    """Strictly validate that the file is a valid PDF."""
    pdf_file = Path(file_path)
    
    # Check if file exists
    if not pdf_file.exists():
        return False, f"File not found: {file_path}"
    
    # Check if file extension is .pdf
    if pdf_file.suffix.lower() != '.pdf':
        return False, f"Invalid file type. Only PDF files are accepted. Got: {pdf_file.suffix}"
    
    # Check if file is readable and is actually a PDF
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False, "File is not a valid PDF (invalid header)"
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    return True, "Valid PDF file"


def is_valid_url(url: str) -> bool:
    """Check if the input is a valid URL."""
    return url.startswith('http://') or url.startswith('https://')


def validate_request_data(data):
    """Validate incoming request data."""
    errors = []
    
    # Check required fields
    if not data.get('source_language') or not isinstance(data.get('source_language'), str):
        errors.append("source_language is required and must be a string")
    
    if not data.get('list_target_languages') or not isinstance(data.get('list_target_languages'), list):
        errors.append("list_target_languages is required and must be a list")
    
    if not data.get('pdf_path') or not isinstance(data.get('pdf_path'), str):
        errors.append("pdf_path is required and must be a string (URL or local path)")
    
    # Check optional fields
    insert_db = data.get('insert_db', False)
    if not isinstance(insert_db, bool):
        if isinstance(insert_db, str):
            insert_db = insert_db.lower() in ['true', '1', 'yes']
        else:
            errors.append("insert_db must be a boolean or string ('true'/'false')")
    
    first_page = data.get('first_page')
    last_page = data.get('last_page')
    
    if first_page is not None and not isinstance(first_page, int):
        errors.append("first_page must be an integer or null")
    
    if last_page is not None and not isinstance(last_page, int):
        errors.append("last_page must be an integer or null")
    
    return errors, insert_db


# ==============================================================================
# MAIN GLOSSARY EXTRACTION FUNCTION
# ==============================================================================

def add_glossary(source_language: str, list_target_languages: list, pdf_path: str, insert_db: bool, first_page=None, last_page=None) -> dict:

    """
    Main glossary extraction function - PDF from URL or Local Path
    WITH PARALLEL PROCESSING FOR FASTER EXTRACTION
    
    Args:
        source_language: Source language (e.g., "english")
        list_target_languages: List of target languages (e.g., ["hindi", "gujarati"])
        pdf_path: URL or local file path to PDF
        insert_db: Boolean to insert into database or not
        first_page: Optional first page number
        last_page: Optional last page number
    
    Returns:
        Dict with format: {"message": "...", "status": "...", "data": [...], "table_name": "..."}
    """
    
    start_time = time.time()  # Track total execution time
    
    print("\n" + "="*60)
    print("GLOSSARY EXTRACTION - PDF (URL or LOCAL)")
    print("WITH PARALLEL PROCESSING")
    print("="*60)
    
    # STEP 1: VALIDATE INPUTS
    print("\n[STEP 1] Validating inputs...")
    
    if not source_language or not list_target_languages:
        return format_error_response("Source and target languages are required")
    
    source_lang_lower = source_language.lower()
    target_langs_lower = [lang.lower() for lang in list_target_languages]
    
    if source_lang_lower in target_langs_lower:
        return format_error_response("Source language cannot be in target languages")
    
    print(f"✓ Source Language: {source_language}")
    print(f"✓ Target Languages: {', '.join(list_target_languages)}")
    
    # STEP 2: VALIDATE PDF FILE
    print("\n[STEP 2] Validating PDF source (URL or LOCAL)...")
    
    local_pdf_path = pdf_path
    download_folder = None
    
    # Check if it's a URL
    if is_valid_url(pdf_path):
        print(f"✓ PDF source detected as URL")
        download_folder = Path(PDF_OUTPUT_FOLDER) / f"downloads_{int(time.time())}"
        
        try:
            local_pdf_path = download_pdf_from_url(pdf_path, str(download_folder))
            print(f"✓ PDF downloaded successfully")
        except Exception as e:
            return format_error_response(f"Failed to download PDF: {str(e)}")
    else:
        # Local file validation
        is_valid, validation_msg = validate_pdf_file(pdf_path)
        
        if not is_valid:
            return format_error_response(f"PDF validation failed: {validation_msg}")
        
        pdf_file = Path(pdf_path)
        print(f"✓ PDF validation passed: {pdf_file.name}")
    
    # STEP 3: INITIALIZE LLM
    print("\n[STEP 3] Initializing LLM...")
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-DUMMYKEY":
        if download_folder:
            shutil.rmtree(download_folder, ignore_errors=True)
        return format_error_response("OPENAI_API_KEY not configured. Set it in your .env file.")
    
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        api_key=OPENAI_API_KEY
    )
    print("✓ LLM initialized")
    
    # STEP 4: EXTRACT IMAGES FROM PDF
    print("\n[STEP 4] Converting PDF to images...")
    
    temp_folder = Path(PDF_OUTPUT_FOLDER) / "temp_images"
    
    extracted_images = extract_images_from_pdf(
        local_pdf_path,
        str(temp_folder),
        PDF_DPI,
        first_page,
        last_page
    )
    
    if not extracted_images:
        if download_folder:
            shutil.rmtree(download_folder, ignore_errors=True)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return format_error_response("Failed to extract images from PDF")
    
    print(f"✓ Extracted {len(extracted_images)} images")
    
    # STEP 5: DETECT AND VALIDATE LANGUAGES
    print("\n[STEP 5] Detecting and validating languages...")
    
    detected_langs = detect_languages_from_raw_image(extracted_images[0], llm)
    
    if not detected_langs:
        if download_folder:
            shutil.rmtree(download_folder, ignore_errors=True)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return format_error_response("Could not detect any languages in PDF")
    
    # VALIDATION: Check if provided languages match detected languages
    is_valid_langs = validate_languages(source_language, list_target_languages, detected_langs)
    
    if not is_valid_langs:
        if download_folder:
            shutil.rmtree(download_folder, ignore_errors=True)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return format_error_response(
            f"Language mismatch! Provided languages do not match PDF content. "
            f"PDF contains: {detected_langs}"
        )
    
    # STEP 6: EXTRACT GLOSSARY FROM ALL IMAGES IN PARALLELl
    print("\n[STEP 6] Extracting glossary entries (PARALLEL PROCESSING)...")
    extraction_start = time.time()
    
    all_entries = extract_glossary_parallel(                                                                                #HERE the extract glossary function is being called
        extracted_images,
        llm,
        source_language,
        list_target_languages,
        max_workers=MAX_WORKERS
    )
    
    extraction_time = time.time() - extraction_start                                               
    
    if not all_entries:
        if download_folder:
            shutil.rmtree(download_folder, ignore_errors=True)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return format_error_response("No glossary entries could be extracted from PDF")
    
    print(f"✓ Total entries extracted: {len(all_entries)}")
    print(f"  Extraction time: {extraction_time:.2f} seconds")
    

    table_name = None
    
    print("\n[STEP 7] Database operation...")
    
    if insert_db:
        print("  Inserting into database...")
        table_name = f"glossary_new_one"
        
        try:
            json_data = [entry.dict() for entry in all_entries]
            if save_to_database(json_data, table_name):
                print(f"  ✓ Database insertion successful in table: {table_name}")
            else:
                print("  ✗ Database insertion failed (but extraction succeeded)")
                table_name = None
        except Exception as e:
            print(f"  ✗ Database error: {e}")
            table_name = None
    else:
        print("  Skipping database insertion (insert_db=False)")
    
    # STEP 8: CLEANUP
    print("\n[STEP 8] Cleaning up temporary files...")
    try:
        # Delete temp_images folder (contains all extracted images)
        if temp_folder and os.path.exists(str(temp_folder)):
            shutil.rmtree(str(temp_folder), ignore_errors=False)
            print("✓ Temporary images folder 'temp_images' deleted successfully")

            # Also delete parent folder if empty
            parent_dir = Path(temp_folder).parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()
                print(f"✓ Parent folder '{parent_dir}' deleted (was empty)")

        # Delete download_folder if it exists (for URL downloads)
        if download_folder and os.path.exists(str(download_folder)):
            shutil.rmtree(str(download_folder), ignore_errors=False)
            print("✓ Downloaded PDF folder deleted successfully")

    except Exception as e:
        print(f"✗ Error while deleting temporary files: {e}")

    
    # STEP 9: FORMAT RESPONSE
    print("\n[STEP 9] Formatting response...")
    
    output_data = [entry.dict() for entry in all_entries]
    
    response = format_success_response(output_data, len(all_entries), table_name)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✓ EXTRACTION COMPLETE - {len(all_entries)} entries")
    if table_name:
        print(f"✓ Data stored in table: {table_name}")
    print(f"✓ Total execution time: {total_time:.2f} seconds")
    print(f"✓ Parallel extraction time: {extraction_time:.2f} seconds")
    print("="*60 + "\n")
    
    return response

# example of some of the pdfs   

# "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/01/2023010657.pdf"  gujarati pdf
# https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/01/2023011933.pdf telugu pdf
# https://www.wicourts.gov/services/interpreter/docs/arabicglossary.pdf for arabic glossary
# https://mncourts.gov/_media/migration/assets/documents/language_access_plans/resources_russian/russian-legal-glossary.pdf for russian glossary
# https://www.wicourts.gov/services/interpreter/docs/frenchglossary.pdf pdf for french glossary
# https://www.lawsect.kerala.gov.in/english/images/pdf/legal-glossary/Legal_Glossary_A_to_D.pdf english to malayalam glossary








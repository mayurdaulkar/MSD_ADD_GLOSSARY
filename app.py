# import config
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import sys
# from pathlib import Path
# # Import from models.py
# from models import ExtractRequest

# # Import from glossary_processor.py
# from glossary_processor import (
#     add_glossary,
#     validate_request_data,
#     format_error_response
# )


# src_path = Path(__file__).parent / "src"
# sys.path.insert(0, str(src_path))
# # ==============================================================================
# # FASTAPI APP SETUP - THIS IS THE ONLY PLACE WHERE APP IS DEFINED
# # ==============================================================================
# app = FastAPI(
#     title="Glossary Extraction API",
#     description="API for extracting and processing glossary entries",
#     version="1.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ==============================================================================
# # API ENDPOINTS - ALL ENDPOINTS GO HERE
# # ==============================================================================

# @app.post("/extract-glossary")
# async def extract_glossary_api(request: ExtractRequest):
#     """FastAPI endpoint for glossary extraction."""
#     try:
#         # Validate request data using function from glossary_processor
#         errors, insert_db = validate_request_data(request.dict())
        
#         if errors:
#             error_msg = "Validation errors: " + "; ".join(errors)
#             raise HTTPException(status_code=400, detail=error_msg)
        
#         # Clean and prepare parameters
#         source_language = request.source_language.lower().strip()
#         list_target_languages = [lang.lower().strip() for lang in request.list_target_languages]
#         pdf_path = request.pdf_path.strip()
#         first_page = request.first_page
#         last_page = request.last_page
        
#         # Validate target languages are not empty
#         if not list_target_languages:
#             raise HTTPException(status_code=400, detail="Target languages are required and cannot be empty")
        
#         # Call the main extraction function from glossary_processor
#         result = add_glossary(
#             source_language=source_language,
#             list_target_languages=list_target_languages,
#             pdf_path=pdf_path,
#             insert_db=insert_db,
#             first_page=first_page,
#             last_page=last_page
#         )
        
#         # Return response
#         if result['status'] == 'Success':
#             return JSONResponse(content=result, status_code=200)
#         else:
#             return JSONResponse(content=result, status_code=400)
    
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         error_response = format_error_response(f"Internal server error: {str(e)}")
#         print(f"✗ API Error: {e}")
#         return JSONResponse(content=error_response, status_code=500)


# @app.get("/health")
# async def health_check():
#     """Health check endpoint - Verifies API is running."""
#     return {
#         "status": "healthy",
#         "message": "Glossary Extraction API is running"
#     }


# @app.get("/")
# async def index():
#     """API information endpoint - Shows available endpoints."""
#     return {
#         "message": "Glossary Extraction API",
#         "version": "1.0",
#         "endpoints": {
#             "POST /extract-glossary": "Extract glossary from PDF",
#             "GET /health": "Health check",
#             "GET /": "API information"
#         },
#         "request_format": {
#             "source_language": "string (required)",
#             "list_target_languages": "array of strings (required)",
#             "pdf_path": "string - URL or local path (required)",
#             "insert_db": "boolean (optional, default: false)",
#             "first_page": "integer or null (optional, default: null)",
#             "last_page": "integer or null (optional, default: null)"
#         },
#         "documentation": {
#             "swagger_ui": "/docs",
#             "redoc": "/redoc"
#         }
#     }


# # ==============================================================================
# # SERVER STARTUP
# # ==============================================================================

# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("GLOSSARY EXTRACTION API - FASTAPI SERVER")
#     print("="*60)
#     print(f"\n✓ Starting server on {config.HOST}:{config.PORT}")
#     print(f"✓ Debug mode: {config.DEBUG}")
#     print(f"\n📌 Available endpoints:")
#     print(f"   GET  http://{config.HOST}:{config.PORT}/docs        - API Documentation (Swagger)")
#     print(f"   GET  http://{config.HOST}:{config.PORT}/redoc       - Alternative Documentation")
#     print(f"   GET  http://{config.HOST}:{config.PORT}/            - API Info")
#     print(f"   GET  http://{config.HOST}:{config.PORT}/health      - Health Check")
#     print(f"   POST http://{config.HOST}:{config.PORT}/extract-glossary - Extract Glossary")
#     print("\n" + "="*60 + "\n")
    
#     uvicorn.run(
#         "app:app",  # This means: from app.py, use the app object
#         host=config.HOST,
#         port=config.PORT,
#         reload=config.DEBUG
#     )








import sys
from pathlib import Path

# Add src directory to path FIRST
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import from the glossary subpackage
from glossary.config import *  # or: import glossary.config as config
from glossary.models import ExtractRequest
from glossary.glossary_processor import (
    add_glossary,
    validate_request_data,
    format_error_response
)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==============================================================================
# FASTAPI APP SETUP - THIS IS THE ONLY PLACE WHERE APP IS DEFINED
# ==============================================================================
app = FastAPI(
    title="Glossary Extraction API",
    description="API for extracting and processing glossary entries",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# API ENDPOINTS - ALL ENDPOINTS GO HERE
# ==============================================================================

@app.post("/extract-glossary")
async def extract_glossary_api(request: ExtractRequest):
    """FastAPI endpoint for glossary extraction."""
    try:
        # Validate request data using function from glossary_processor
        errors, insert_db = validate_request_data(request.dict())
        
        if errors:
            error_msg = "Validation errors: " + "; ".join(errors)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Clean and prepare parameters
        source_language = request.source_language.lower().strip()
        list_target_languages = [lang.lower().strip() for lang in request.list_target_languages]
        pdf_path = request.pdf_path.strip()
        first_page = request.first_page
        last_page = request.last_page
        
        # Validate target languages are not empty
        if not list_target_languages:
            raise HTTPException(status_code=400, detail="Target languages are required and cannot be empty")
        
        # Call the main extraction function from glossary_processor
        result = add_glossary(
            source_language=source_language,
            list_target_languages=list_target_languages,
            pdf_path=pdf_path,
            insert_db=insert_db,
            first_page=first_page,
            last_page=last_page
        )
        
        # Return response
        if result['status'] == 'Success':
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        error_response = format_error_response(f"Internal server error: {str(e)}")
        print(f"✗ API Error: {e}")
        return JSONResponse(content=error_response, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint - Verifies API is running."""
    return {
        "status": "healthy",
        "message": "Glossary Extraction API is running"
    }


@app.get("/")
async def index():
    """API information endpoint - Shows available endpoints."""
    return {
        "message": "Glossary Extraction API",
        "version": "1.0",
        "endpoints": {
            "POST /extract-glossary": "Extract glossary from PDF",
            "GET /health": "Health check",
            "GET /": "API information"
        },
        "request_format": {
            "source_language": "string (required)",
            "list_target_languages": "array of strings (required)",
            "pdf_path": "string - URL or local path (required)",
            "insert_db": "boolean (optional, default: false)",
            "first_page": "integer or null (optional, default: null)",
            "last_page": "integer or null (optional, default: null)"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }


# ==============================================================================
# SERVER STARTUP
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("GLOSSARY EXTRACTION API - FASTAPI SERVER")
    print("="*60)
    print(f"\n✓ Starting server on {config.HOST}:{config.PORT}")
    print(f"✓ Debug mode: {config.DEBUG}")
    print(f"\n📚 Available endpoints:")
    print(f"   GET  http://{config.HOST}:{config.PORT}/docs        - API Documentation (Swagger)")
    print(f"   GET  http://{config.HOST}:{config.PORT}/redoc       - Alternative Documentation")
    print(f"   GET  http://{config.HOST}:{config.PORT}/            - API Info")
    print(f"   GET  http://{config.HOST}:{config.PORT}/health      - Health Check")
    print(f"   POST http://{config.HOST}:{config.PORT}/extract-glossary - Extract Glossary")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
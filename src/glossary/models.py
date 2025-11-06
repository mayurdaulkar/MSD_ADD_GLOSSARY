"""
Models Module
Contains all Pydantic data models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class GlossaryEntry(BaseModel):
    """Pydantic model for a single glossary entry"""
    source_language: str = Field(description="The source language, e.g. 'english'")
    target_language: str = Field(description="The target language, e.g. 'hindi'")
    source_text: str = Field(description="The original text in the source language")
    translated_text: str = Field(description="The translated version of the source text")
    legal_reference: str = Field(description="The legal reference section or clause")

    class Config:
        json_schema_extra = {
            "example": {
                "source_language": "english",
                "target_language": "hindi",
                "source_text": "contract",
                "translated_text": "अनुबंध",
                "legal_reference": "1.1"
            }
        }


class GlossaryExtraction(BaseModel):
    """Pydantic model for glossary extraction response"""
    rows: List[GlossaryEntry]


class ExtractRequest(BaseModel):
    """Pydantic model for request validation"""
    source_language: str = Field(description="Source language (e.g., 'english')")
    list_target_languages: List[str] = Field(description="List of target languages (e.g., ['hindi', 'gujarati'])")
    pdf_path: str = Field(description="URL or local file path to PDF")
    insert_db: Optional[bool] = Field(default=False, description="Whether to insert extracted data into database")
    first_page: Optional[int] = Field(default=None, description="First page number to process (optional)")
    last_page: Optional[int] = Field(default=None, description="Last page number to process (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "source_language": "english",
                "list_target_languages": ["hindi", "gujarati"],
                "pdf_path": "https://example.com/glossary.pdf",
                "insert_db": True,
                "first_page": 1,
                "last_page": None
            }
        }
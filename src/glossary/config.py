# config for the all in one file
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env FILE
load_dotenv()

# ==============================================================================
# CONFIG CLASS
# ==============================================================================

class Config:
    """
    Configuration class for Glossary Extraction API
    Loads all settings from environment variables or uses defaults
    """
    
    # ========== LLM Configuration ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-DUMMYKEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    # ========== Database Configuration ==========
    DB_CONFIG = {
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5432"),
        'name': os.getenv("DB_NAME", "glossary_db"),
        'user': os.getenv("DB_USER", "postgres"),
        'password': os.getenv("DB_PASSWORD", "postgres"),
    }

    # ========== PDF Proccessing configuration =========
    PDF_DPI = int(os.getenv("PDF_DPI", "300"))
    PDF_THREAD_COUNT = int(os.getenv("PDF_THREAD_COUNT", "4"))
    # Create absolute path and ensure folder exists
    PDF_OUTPUT_FOLDER = os.path.abspath(os.getenv("PDF_OUTPUT_FOLDER", "temp_images"))
    PDF_FORMAT = os.getenv("PDF_FORMAT", "jpeg")
    
    # ========== Image Enhancement Configuration ==========
    IMAGE_CONTRAST = float(os.getenv("IMAGE_CONTRAST", "1.5"))
    IMAGE_SHARPNESS = float(os.getenv("IMAGE_SHARPNESS", "1.5"))
    IMAGE_BRIGHTNESS = float(os.getenv("IMAGE_BRIGHTNESS", "1.2"))
    IMAGE_MEDIAN_FILTER_SIZE = int(os.getenv("IMAGE_MEDIAN_FILTER_SIZE", "3"))

    # ========== Flask Configuration ==========
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    
    # ========== Application Configuration ==========
    OUTPUT_JSON_FILE = os.getenv("OUTPUT_JSON_FILE", "extracted_glossary.json")


# ==============================================================================
# INSTANTIATE CONFIG
# ==============================================================================

config = Config()

# Create PDF_OUTPUT_FOLDER if it doesn't exist
Path(config.PDF_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# EXPORT CONFIGURATION VARIABLES
# ==============================================================================

# LLM Settings
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL = config.OPENAI_MODEL
OPENAI_MAX_TOKENS = config.OPENAI_MAX_TOKENS
OPENAI_TEMPERATURE = config.OPENAI_TEMPERATURE

# Database Settings
DB_CONFIG = config.DB_CONFIG

# PDF Processing Settings
PDF_DPI = config.PDF_DPI
PDF_THREAD_COUNT = config.PDF_THREAD_COUNT
PDF_OUTPUT_FOLDER = config.PDF_OUTPUT_FOLDER
PDF_FORMAT = config.PDF_FORMAT

# Image Enhancement Settings
IMAGE_CONTRAST = config.IMAGE_CONTRAST
IMAGE_SHARPNESS = config.IMAGE_SHARPNESS
IMAGE_BRIGHTNESS = config.IMAGE_BRIGHTNESS
IMAGE_MEDIAN_FILTER_SIZE = config.IMAGE_MEDIAN_FILTER_SIZE

# Flask Settings
DEBUG = config.DEBUG
PORT = config.PORT
HOST = config.HOST

# Application Settings
OUTPUT_JSON_FILE = config.OUTPUT_JSON_FILE









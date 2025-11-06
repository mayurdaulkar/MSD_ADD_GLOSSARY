# prompt.py
# This file contains all LLM prompts used in the glossary extraction system

# ==============================================================================
# GLOSSARY EXTRACTION PROMPTS
# ==============================================================================

# Prompt 1: Extract glossary entries WITH specified languages
GLOSSARY_WITH_LANGUAGES = """
Extract all glossary rows from this table image.
Source language: {source_lang}
Target languages: {target_langs}

For each row, return these fields:
- source_language
- target_language
- source_text
- translated_text
- legal_reference
"""

# Prompt 2: Extract glossary entries WITHOUT specified languages (auto-detect)
GLOSSARY_WITHOUT_LANGUAGES = """
Extract all glossary rows from this table image.
Identify the source language and all target languages present.

For each row, return these fields:
- source_language
- target_language
- source_text
- translated_text
- legal_reference
"""

# Prompt 3: Extract raw text from image for language detection
RAW_TEXT_EXTRACTION = """Extract ALL text from this image, including headers, table content, and any visible text. Return the complete text as a single string."""

# Prompt 4: Language detection from raw text
LANGUAGE_DETECTION = """CRITICAL LANGUAGE DETECTION TASK:

I have extracted text from a legal glossary table. You MUST identify ALL languages present.

EXTRACTED TEXT:
{raw_text}

STRICT REQUIREMENTS:
1. Identify EVERY language present in this text
2. Look for these specific languages: english, hindi, marathi, gujarati, bengali, punjabi, tamil, telugu, kannada, malayalam, 
    odia, urdu, assamese, konkani, manipuri, sanskrit, bodo, santali, dogri, nepali, kashmiri,
    maithili, chinese, japanese, korean, thai, vietnamese, burmese, khmer, lao, mongolian, arabic, persian, farsi, turkish,
    hebrew, kurdish, pashto, uzbek, french, german, spanish, italian, portuguese, russian, ukrainian, polish, romanian, greek,
    czech, hungarian, dutch, swedish, norwegian, finnish, danish, icelandic, serbian, croatian, bosnian, slovak, bulgarian, albanian, latvian, lithuanian,
    estonian, swahili, amharic, zulu, hausa, somali, yoruba, igbo, afrikaans, quechua, guarani, nahuatl, mapudungun, maori,
    samoan, tongan, fijian, hawaiian, latin, esperanto, catalan, basque, galician, slovenian     
           
3. Pay attention to:
    - Column headers that mention languages
    - Different scripts (Latin, Devanagari, Telugu, Tamil, etc.)
    - Language names mentioned in the text
4. If you see English text AND any non-English text, include both
5. BE THOROUGH - don't miss any languages

RETURN FORMAT: 
ONLY a JSON array of language names in lowercase, like: ["english", "hindi", "telugu"]

DO NOT return anything else. No explanations, no markdown."""
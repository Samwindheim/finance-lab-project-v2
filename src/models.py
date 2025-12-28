"""
This file defines the Pydantic models used for data validation across the extraction pipeline.
"""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, RootModel

# --- Extraction Definitions (src/extraction_definitions.json) ---

class ExtractionDefinition(BaseModel):
    description: str
    source_types: List[str]
    semantic_search_query: str
    page_selection_strategy: str = "consecutive"

class ExtractionDefinitions(RootModel):
    root: Dict[str, ExtractionDefinition]

# --- LLM Extraction Outputs ---

class Investor(BaseModel):
    name: str
    amount_in_cash: Optional[Union[float, str]] = None
    amount_in_percentage: Optional[Union[float, str]] = None
    level: Optional[int] = None

class ImportantDates(BaseModel):
    record_date: Optional[str] = None
    sub_start_date: Optional[str] = None
    sub_end_date: Optional[str] = None
    inc_rights_date: Optional[str] = None
    ex_rights_date: Optional[str] = None
    rights_start_date: Optional[str] = None
    rights_end_date: Optional[str] = None

class OfferingTerms(BaseModel):
    shares_required: Optional[int] = None
    rights_received: Optional[int] = None
    rights_required: Optional[int] = None
    units_received: Optional[int] = None
    shares_in_unit: Optional[int] = None
    unit_sub_price: Optional[Union[float, str]] = None
    offered_units: Optional[Union[int, str]] = None

class ExtractionResult(BaseModel):
    investors: Optional[List[Investor]] = None
    important_dates: Optional[ImportantDates] = None
    offering_terms: Optional[OfferingTerms] = None
    # Add other fields as needed, or use an extra field for flexibility
    # For now, we'll allow extra fields since the definitions are dynamic
    class Config:
        extra = "allow"

# --- Final Output Structure (output_json/ISSUE_ID_extraction.json) ---

class DocumentEntry(BaseModel):
    issue_id: str
    id: Optional[str] = None
    investors: Optional[List[Investor]] = None
    important_dates: Optional[ImportantDates] = None
    offering_terms: Optional[OfferingTerms] = None

    class Config:
        extra = "allow"

class FinalOutput(RootModel):
    root: Dict[str, DocumentEntry]


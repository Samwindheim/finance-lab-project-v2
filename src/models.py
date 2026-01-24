"""
This file defines the Pydantic models used for data validation across the extraction pipeline.
"""

from typing import List, Optional, Dict, Union, get_args, get_origin
from pydantic import BaseModel, Field, RootModel

# From database table issues
"""
general_meeting_date, inc_rights_date, ex_rights_date, record_date, rights_start_date, rights_end_date, sub_start_date, sub_end_date, ipo_trading_date,

minimum_sub_condition, shares_required, rights_received, rights_required, units_received, shares_in_unit, unit_extra_content, unit_sub_price, outstanding_shares, offered_units, offered_units_unlisted, earnout, abb_target, approx_total_commitment, approx_commitment, approx_guarantor, over_allotment_size, 

isin_rights, isin_units, noncash_consideration, unit_sub_total, unit_sub_with_rights, unit_sub_without_rights, unit_sub_guarantor, unit_pct_sub_with_rights, unit_pct_sub_without_rights, unit_pct_sub_guarantor, over_allotment_allocation, secondary_offering, secondary_sub, 

ipo_interval_low, ipo_interval_high, over_allotment_size_secondary, over_allotment_allocation_secondary, 

no_financial_advisor, no_issuing_agent, no_legal_advisor, no_over_allotment, no_terms_initially, no_secondary_offering, no_guarantors, no_isin_units, no_participants_named, investors_after_terms, investors_after_prospectus, rto, published_at, created_at, updated_at, no_investors_named
"""

# --- Extraction Definitions (src/extraction_definitions.json) ---

class ExtractionDefinition(BaseModel):
    description: str
    source_types: List[str]
    issue_types: Optional[List[str]] = None
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
    general_meeting_date: Optional[str] = None
    ipo_trading_date: Optional[str] = None

class OfferingTerms(BaseModel):
    shares_required: Optional[int] = None
    rights_received: Optional[int] = None
    rights_required: Optional[int] = None
    units_received: Optional[int] = None
    shares_in_unit: Optional[int] = None
    unit_sub_price: Optional[Union[float, str]] = None
    offered_units: Optional[Union[int, str]] = None
    # From IPO
    secondary_offering: Optional[Union[int, str]] = None
    over_allotment_size_secondary: Optional[Union[int, str]] = None

class OfferingOutcome(BaseModel):
    unit_sub_total: Optional[Union[int, str]] = None
    unit_pct_sub_with_rights: Optional[Union[float, str]] = None
    unit_pct_sub_without_rights: Optional[Union[float, str]] = None
    unit_pct_sub_guarantor: Optional[Union[float, str]] = None
    unit_sub_with_rights: Optional[Union[int, str]] = None
    unit_sub_without_rights: Optional[Union[int, str]] = None
    unit_sub_guarantor: Optional[Union[int, str]] = None
    # From IPO
    secondary_sub: Optional[Union[int, str]] = None

class GeneralInfo(BaseModel):
    finansiella_radgivare: Optional[str] = None
    legalradgivare: Optional[str] = None
    emissionsinstitut: Optional[str] = None

class ExtractionResult(BaseModel):
    investors: Optional[List[Investor]] = None
    important_dates: Optional[ImportantDates] = None
    offering_terms: Optional[OfferingTerms] = None
    offering_outcome: Optional[OfferingOutcome] = None
    # General info is not in issue db so removing for testing, isin
    # data isn't extracting reliably.
    # general_info: Optional[GeneralInfo] = None
    # isin_units: Optional[str] = None
    # isin_rights: Optional[str] = None

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
    offering_outcome: Optional[OfferingOutcome] = None
    # General info is not in issue db so removing for testing, isin
    # data isn't extracting reliably.
    # general_info: Optional[GeneralInfo] = None
    # isin_units: Optional[str] = None
    # isin_rights: Optional[str] = None

    class Config:
        extra = "allow"

class FinalOutput(RootModel):
    root: Dict[str, DocumentEntry]

def get_fields_for_category(category: str) -> List[str]:
    """Dynamically retrieves field names for a given category from ExtractionResult."""
    field_info = ExtractionResult.model_fields.get(category)
    if not field_info:
        return []
    
    # Unwrap Optional/Union to get the actual type
    annotation = field_info.annotation
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        # Find the one that isn't NoneType
        annotation = next((a for a in args if a is not type(None)), None)

    # If it's a List[Investor], get the Investor fields
    if get_origin(annotation) is list:
        annotation = get_args(annotation)[0]

    # If it's a Pydantic model, return its fields
    if hasattr(annotation, "model_fields"):
        return list(annotation.model_fields.keys())
    
    # Otherwise it's a single field (like isin_units)
    return [category]

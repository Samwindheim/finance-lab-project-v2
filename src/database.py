import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()

class FinanceDB:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in .env")
        self.engine = create_engine(self.db_url)

    def get_issue_data(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Fetch all terms, dates, and outcomes for an issue from the 'issues' table."""
        query = f"SELECT * FROM issues WHERE id = '{issue_id}'"
        df = pd.read_sql(query, self.engine)
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_investors_data(self, issue_id: str) -> pd.DataFrame:
        """Fetch individual investor commitments with names (using aliases if available) for an issue."""
        query = f"""
            SELECT 
                COALESCE(ia.name_alias, i.name) as name,
                ii.amount_in_cash,
                ii.amount_in_percentage,
                ii.fee_level as level
            FROM investors_issues ii
            JOIN investors i ON ii.investor_id = i.id
            LEFT JOIN investor_aliases ia ON ii.investor_alias_id = ia.id
            WHERE ii.issue_id = '{issue_id}'
        """
        return pd.read_sql(query, self.engine)

    def get_all_sources(self) -> pd.DataFrame:
        """Fetch all source documents from the database."""
        query = "SELECT * FROM sources"
        return pd.read_sql(query, self.engine)

    def find_sources_by_issue(self, issue_id: str) -> pd.DataFrame:
        """Fetch all source documents for a specific issue."""
        query = f"SELECT * FROM sources WHERE issue_id = '{issue_id}'"
        return pd.read_sql(query, self.engine)

    def find_source_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Find a source document by its URL."""
        query = text("SELECT * FROM sources WHERE source_url = :url")
        df = pd.read_sql(query, self.engine, params={"url": url})
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def find_source_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Find a source document by its unique ID."""
        query = text("SELECT * FROM sources WHERE id = :id")
        df = pd.read_sql(query, self.engine, params={"id": doc_id})
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def save_ai_extraction(self, issue_id: Optional[str], doc_id: Optional[str], extraction_field: str, data: Dict[str, Any], source_url: Optional[str] = None):
        """
        Saves an AI extraction to the 'ai_extractions' table.
        Creates the table if it doesn't exist.
        Uses UPSERT logic to update existing entries for the same source_url and field.
        """
        if not source_url:
            logger.error("Cannot save AI extraction: source_url is missing.")
            return

        # Ensure the table exists (Using MySQL compatible syntax)
        # source_url and extraction_field are the unique identifiers for a draft.
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ai_extractions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            issue_id VARCHAR(255) NULL,
            doc_id VARCHAR(255) NULL,
            source_url VARCHAR(767) NOT NULL,
            extraction_field VARCHAR(255) NOT NULL,
            data JSON,
            status VARCHAR(50) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_url_field (source_url, extraction_field)
        );
        """
        
        # MySQL UPSERT (INSERT ... ON DUPLICATE KEY UPDATE)
        insert_query = text("""
        INSERT INTO ai_extractions (issue_id, doc_id, source_url, extraction_field, data, status)
        VALUES (:issue_id, :doc_id, :source_url, :extraction_field, :data, 'pending')
        ON DUPLICATE KEY UPDATE
            issue_id = VALUES(issue_id),
            doc_id = VALUES(doc_id),
            data = VALUES(data),
            status = 'pending',
            updated_at = CURRENT_TIMESTAMP
        """)

        with self.engine.begin() as conn:
            conn.execute(text(create_table_query))
            
            # Migration: Drop old unique keys if they exist
            for old_key in ["unique_doc_field", "unique_anchor_field", "unique_doc_url_field"]:
                try:
                    conn.execute(text(f"ALTER TABLE ai_extractions DROP INDEX {old_key}"))
                except Exception:
                    pass
            
            # Migration: Clean up duplicates based on URL and Field
            # We keep the entry with the highest ID (most recent)
            cleanup_query = """
            DELETE t1 FROM ai_extractions t1
            INNER JOIN ai_extractions t2 
            WHERE t1.id < t2.id 
            AND t1.source_url = t2.source_url 
            AND t1.extraction_field = t2.extraction_field;
            """
            try:
                conn.execute(text(cleanup_query))
            except Exception as e:
                logger.warning(f"Failed to cleanup duplicates: {e}")

            # Migration: Ensure source_url is NOT NULL for the unique key
            try:
                conn.execute(text("ALTER TABLE ai_extractions MODIFY COLUMN source_url VARCHAR(767) NOT NULL"))
            except Exception:
                pass

            # Migration: Ensure doc_id is NULLable (fix for previous NOT NULL DEFAULT '')
            try:
                conn.execute(text("ALTER TABLE ai_extractions MODIFY COLUMN doc_id VARCHAR(255) NULL"))
            except Exception:
                pass

            # Migration: Add new unique key if it doesn't exist
            try:
                conn.execute(text("ALTER TABLE ai_extractions ADD UNIQUE KEY unique_url_field (source_url, extraction_field)"))
            except Exception:
                pass
            
            # Migration: Remove source_anchor column if it exists
            try:
                conn.execute(text("ALTER TABLE ai_extractions DROP COLUMN source_anchor"))
            except Exception:
                pass

            conn.execute(insert_query, {
                "issue_id": issue_id,
                "doc_id": doc_id,
                "source_url": source_url,
                "extraction_field": extraction_field,
                "data": json.dumps(data)
            })

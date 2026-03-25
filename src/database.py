"""
Database Interface.

Handles SQL connections and operations for the issues, sources, and ai_extractions tables.
Includes automated schema setup and migration logic for the staging area.
"""
import os
import json
from typing import Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from logger import setup_logger

logger = setup_logger(__name__)

class FinanceDB:
    _schema_ready = False

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.engine = create_engine(self.db_url) if self.db_url else None

    def _ensure_schema(self):
        """Creates the ai_extractions table and runs one-time migrations if not already done."""
        if not self.engine:
            raise ValueError("DATABASE_URL is not configured.")
        if FinanceDB._schema_ready:
            return
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
        with self.engine.begin() as conn:
            conn.execute(text(create_table_query))
            for old_key in ["unique_doc_field", "unique_anchor_field", "unique_doc_url_field"]:
                try:
                    conn.execute(text(f"ALTER TABLE ai_extractions DROP INDEX {old_key}"))
                except Exception:
                    pass
            try:
                conn.execute(text("ALTER TABLE ai_extractions MODIFY COLUMN source_url VARCHAR(767) NOT NULL"))
                conn.execute(text("ALTER TABLE ai_extractions MODIFY COLUMN extraction_field VARCHAR(255) NOT NULL"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE ai_extractions MODIFY COLUMN doc_id VARCHAR(255) NULL"))
            except Exception:
                pass
            cleanup_query = """
            DELETE t1 FROM ai_extractions t1
            INNER JOIN ai_extractions t2
            ON t1.source_url = t2.source_url
            AND t1.extraction_field = t2.extraction_field
            WHERE t1.id < t2.id;
            """
            try:
                conn.execute(text(cleanup_query))
                conn.execute(text("ALTER TABLE ai_extractions ADD UNIQUE KEY unique_url_field (source_url(512), extraction_field)"))
            except Exception as e:
                logger.debug(f"Migration step (unique key) skipped or failed: {e}")
            try:
                conn.execute(text("ALTER TABLE ai_extractions DROP COLUMN source_anchor"))
            except Exception:
                pass
        FinanceDB._schema_ready = True

    def get_issue_data(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Fetch all terms, dates, and outcomes for an issue from the 'issues' table."""
        query = text("SELECT * FROM issues WHERE id = :issue_id")
        df = pd.read_sql(query, self.engine, params={"issue_id": issue_id})
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_investors_data(self, issue_id: str) -> pd.DataFrame:
        """Fetch individual investor commitments with names (using aliases if available) for an issue."""
        query = text("""
            SELECT
                COALESCE(ia.name_alias, i.name) as name,
                ii.amount_in_cash,
                ii.amount_in_percentage,
                ii.fee_level as level
            FROM investors_issues ii
            JOIN investors i ON ii.investor_id = i.id
            LEFT JOIN investor_aliases ia ON ii.investor_alias_id = ia.id
            WHERE ii.issue_id = :issue_id
        """)
        return pd.read_sql(query, self.engine, params={"issue_id": issue_id})

    def find_sources_by_issue(self, issue_id: str) -> pd.DataFrame:
        """Fetch all source documents for a specific issue."""
        query = text("SELECT * FROM sources WHERE issue_id = :issue_id")
        return pd.read_sql(query, self.engine, params={"issue_id": issue_id})

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
        Saves an AI extraction to the 'ai_extractions' table using UPSERT logic
        keyed on (source_url, extraction_field).
        """
        if not source_url:
            logger.error("Cannot save AI extraction: source_url is missing.")
            return

        self._ensure_schema()

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
            conn.execute(insert_query, {
                "issue_id": issue_id,
                "doc_id": doc_id,
                "source_url": source_url,
                "extraction_field": extraction_field,
                "data": json.dumps(data)
            })

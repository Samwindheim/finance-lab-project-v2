import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Optional, Dict, Any

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

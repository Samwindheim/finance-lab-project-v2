import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("DATABASE_URL not found in .env")
    exit(1)

engine = create_engine(db_url)

try:
    inspector = inspect(engine)
    
    # Check tables in 'labbet'
    print("--- Tables in labbet ---")
    tables = inspector.get_table_names(schema='labbet')
    if not tables:
        # Maybe it's not a schema but just the database name in the URL
        tables = inspector.get_table_names()
    
    for table in tables:
        print(f"\nTable: {table}")
        columns = inspector.get_columns(table, schema='labbet' if 'labbet' in db_url else None)
        for column in columns:
            print(f"  - {column['name']} ({column['type']})")

except Exception as e:
    print(f"Error: {e}")

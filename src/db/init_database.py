from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def init_database():
    """Initialize the database with sample data"""
    # Read database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if db_url is None:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    # Create engine for initial connection to PostgreSQL
    engine = create_engine(db_url)
    
    try:
        # Read SQL script
        sql_path = Path(__file__).parent / "init_db.sql"
        with open(sql_path, 'r') as f:
            sql_script = f.read()
        
        # Execute script
        with engine.connect() as conn:
            conn.execute(text(sql_script))
            conn.commit()
            
        print("Database initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_database()

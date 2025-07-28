from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def test_database_setup():
    """Test database setup and print results"""
    # Create engine
    db_url = os.getenv("DATABASE_URL")
    if db_url is None:
        raise ValueError("DATABASE_URL environment variable is not set.")
    engine = create_engine(db_url)
    
    try:
        # Test queries
        queries = {
            "Department Count": """
                SELECT COUNT(*) as department_count 
                FROM departments;
            """,
            
            "Employee Count": """
                SELECT d.name as department, COUNT(e.id) as employee_count
                FROM departments d
                LEFT JOIN employees e ON d.id = e.department_id
                GROUP BY d.name
                ORDER BY d.name;
            """,
            
            "Project Statistics": """
                SELECT 
                    d.name as department,
                    COUNT(p.id) as project_count,
                    SUM(p.budget) as total_budget
                FROM departments d
                LEFT JOIN projects p ON d.id = p.department_id
                GROUP BY d.name
                ORDER BY d.name;
            """,
            
            "Financial Summary": """
                SELECT 
                    type,
                    COUNT(*) as transaction_count,
                    SUM(amount) as total_amount
                FROM financial_transactions
                GROUP BY type
                ORDER BY type;
            """,
            
            "Project Assignments": """
                SELECT 
                    p.name as project_name,
                    COUNT(pa.id) as assigned_employees,
                    STRING_AGG(DISTINCT pa.role, ', ') as roles
                FROM projects p
                LEFT JOIN project_assignments pa ON p.id = pa.project_id
                GROUP BY p.name
                ORDER BY p.name;
            """
        }
        
        print("Database Connection Test: SUCCESS\n")
        print("=== Database Statistics ===\n")
        
        # Execute and display results for each query
        for title, query in queries.items():
            print(f"\n--- {title} ---")
            df = pd.read_sql_query(query, engine)
            print(df.to_string(index=False))
            print("\n" + "="*50)
        
        # Test data integrity
        integrity_checks = {
            "Orphaned Employees": """
                SELECT COUNT(*) as orphaned_count
                FROM employees e
                LEFT JOIN departments d ON e.department_id = d.id
                WHERE d.id IS NULL;
            """,
            
            "Invalid Project Dates": """
                SELECT COUNT(*) as invalid_count
                FROM projects
                WHERE start_date > end_date;
            """,
            
            "Invalid Allocations": """
                SELECT COUNT(*) as invalid_count
                FROM project_assignments
                WHERE allocation_percentage <= 0 OR allocation_percentage > 100;
            """
        }
        
        print("\n=== Data Integrity Checks ===\n")
        for check_name, check_query in integrity_checks.items():
            df = pd.read_sql_query(check_query, engine)
            result = df.iloc[0, 0]
            status = "✓ PASS" if result == 0 else "✗ FAIL"
            print(f"{check_name}: {status} ({result} issues found)")
        
    except Exception as e:
        print(f"Database Test Failed: {str(e)}")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    test_database_setup()

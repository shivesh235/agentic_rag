import subprocess
import sys
import platform
import os
from pathlib import Path

def check_postgres_installation():
    """Check if PostgreSQL is installed and accessible"""
    print("Checking PostgreSQL installation...")
    
    # Check if we're on Windows
    is_windows = platform.system().lower() == "windows"
    
    if is_windows:
        # Common PostgreSQL installation paths on Windows
        postgres_paths = [
            r"C:\Program Files\PostgreSQL\*\bin",
            r"C:\Program Files (x86)\PostgreSQL\*\bin"
        ]
        
        psql_path = None
        for path in postgres_paths:
            try:
                # Use glob to find the actual version folder
                import glob
                matches = glob.glob(path)
                if matches:
                    psql_path = str(Path(matches[0]) / "psql.exe")
                    if Path(psql_path).exists():
                        break
            except Exception:
                continue
        
        if not psql_path:
            print("PostgreSQL not found in common locations!")
            print("Please install PostgreSQL from: https://www.postgresql.org/download/windows/")
            return False
        
        # Add PostgreSQL bin directory to PATH
        os.environ["PATH"] = str(Path(psql_path).parent) + os.pathsep + os.environ["PATH"]
    
    try:
        # Try to run psql --version
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"PostgreSQL is installed: {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"Error checking PostgreSQL: {str(e)}")
    
    return False

def setup_database():
    """Setup the database using psql"""
    print("\nSetting up database...")
    
    try:
        # Create database
        subprocess.run(
            ["psql", "-U", "postgres", "-c", "CREATE DATABASE ragdb;"],
            check=True,
            capture_output=True,
            text=True
        )
        print("Database 'ragdb' created successfully!")
        
        # Run the initialization script
        init_script = Path(__file__).parent / "init_db.sql"
        subprocess.run(
            ["psql", "-U", "postgres", "-d", "ragdb", "-f", str(init_script)],
            check=True,
            capture_output=True,
            text=True
        )
        print("Database initialized with sample data!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up database: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if not check_postgres_installation():
        print("\nPlease install PostgreSQL and make sure it's accessible from the command line.")
        print("Download from: https://www.postgresql.org/download/")
        sys.exit(1)
    
    if setup_database():
        print("\nDatabase setup completed successfully!")
        print("You can now run test_database.py to verify the setup.")
    else:
        print("\nDatabase setup failed. Please check the error messages above.")

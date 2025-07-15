# create_db.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import ProgrammingError, OperationalError # Import specific errors

print("DEBUG: create_db.py script started.")

# Initialize Flask app (minimal setup just for db context)
app = Flask(__name__)
print("DEBUG: Flask app initialized in create_db.py.")

# Database configuration - MUST match app.py
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    print(f"DEBUG: Using PostgreSQL database: {app.config['SQLALCHEMY_DATABASE_URI']} in create_db.py.")
else:
    # This path should ideally not be hit on Render, but good for local testing
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    print("DEBUG: Using SQLite database for local testing in create_db.py.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
print("DEBUG: SQLAlchemy initialized in create_db.py.")

try:
    # Import your models so SQLAlchemy knows about them
    # Ensure these imports match your app.py model definitions
    # This import line is crucial for db.create_all() to know about your models
    from app import User, Patient, Slide 
    print("DEBUG: Models (User, Patient, Slide) imported successfully in create_db.py.")

    # Create tables within the application context
    with app.app_context():
        print("DEBUG: Entering Flask application context in create_db.py.")
        db.create_all()
        print("DEBUG: Database tables checked/created successfully by create_db.py.")

except (ProgrammingError, OperationalError) as e:
    print(f"ERROR: Database connection or table creation failed in create_db.py: {e}")
    print("This often means:")
    print("  - The DATABASE_URL environment variable is incorrect or missing.")
    print("  - The PostgreSQL database itself is not running or accessible.")
    print("  - There's a firewall blocking the connection.")
    print("  - The database user/password are incorrect.")
    # Exit with a non-zero status to indicate failure to Render
    exit(1)
except ImportError as e:
    print(f"ERROR: Failed to import models from app.py in create_db.py: {e}")
    print("Ensure User, Patient, Slide classes are correctly defined in app.py and app.py is in the root directory.")
    exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during table creation in create_db.py: {e}")
    # Exit with a non-zero status to indicate failure to Render
    exit(1)

print("DEBUG: create_db.py script finished successfully.")


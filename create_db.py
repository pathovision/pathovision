# create_db.py
import os
from app import app
from models import db, User, Patient, Slide # Import your models

# Ensure the app context is available for database operations
with app.app_context():
    # Drop all existing tables (use with caution, this deletes all data!)
    # Only uncomment db.drop_all() if you are absolutely sure you want to clear all data
    # db.drop_all() 
    
    # Create all tables defined in your models
    db.create_all()
    print("Database tables created or already exist.")


# create_db.py
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app (minimal setup just for db context)
app = Flask(__name__)

# Database configuration - MUST match app.py
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    # This path should ideally not be hit on Render, but good for local testing
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Import your models so SQLAlchemy knows about them
# Ensure these imports match your app.py model definitions
from app import User, Patient, Slide # Assuming User, Patient, Slide are defined in app.py

# Create tables within the application context
with app.app_context():
    db.create_all()
    print("DEBUG: Database tables checked/created successfully by create_db.py.")


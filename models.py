# models.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize SQLAlchemy here, but it will be bound to the app in app.py
# This allows models to be defined without a full app context initially
db = SQLAlchemy()

# --- Database Model: User ---
class User(db.Model):
    __tablename__ = 'user' # Explicitly set table name for PostgreSQL case sensitivity
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    patients = db.relationship('Patient', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(password, self.password_hash)

    def __repr__(self):
        return f'<User {self.email}>'

# --- Database Model: Patient ---
class Patient(db.Model):
    __tablename__ = 'patient' # Explicitly set table name
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    slides = db.relationship('Slide', backref='patient', lazy=True)
    diagnostic_report = db.Column(db.Text, nullable=True)
    is_archived = db.Column(db.Boolean, default=False, nullable=False)

    def __repr__(self):
        return f'<Patient {self.name} (User: {self.user_id})>'

# --- Database Model: Slide ---
class Slide(db.Model):
    __tablename__ = 'slide' # Explicitly set table name
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    processed_filename = db.Column(db.String(255), nullable=True)
    biopsy_coords = db.Column(db.Text, nullable=True)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    def __repr__(self):
        return f'<Slide {self.filename} (Patient: {self.patient_id})>'

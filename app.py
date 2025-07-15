from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from PIL import Image, ImageDraw # For drawing on uploaded slides
import base64 # For decoding base64 image data from frontend
import glob # For listing files for cleanup
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

# --- Configuration ---
# IMPORTANT: Use an environment variable for SECRET_KEY in production!
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24)) 

# Database configuration - Use PostgreSQL for Render, fallback to SQLite for local
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # For PostgreSQL on Render
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    print(f"DEBUG: Using PostgreSQL database: {app.config['SQLALCHEMY_DATABASE_URI']}")
else:
    # For local SQLite development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    print("DEBUG: Using SQLite database for local development.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Configuration for general file uploads ---
UPLOAD_FOLDER = 'static/uploads' # Directory to save uploaded images (initial slide)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} # Allowed image file types
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Create upload directory if it doesn't exist

# --- Configuration for Processed Images (Final Stitched Biopsy) ---
PROCESSED_FOLDER = 'static/processed' # Directory to save processed images (including final stitched biopsy images)
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
os.makedirs(PROCESSED_FOLDER, exist_ok=True) # Ensure this folder exists

# --- Configuration for Biopsy Camera Captures (individual tiles) ---
BIOPY_CAPTURE_FOLDER = 'static/biopsy_captures' # Directory to save individual camera frames
app.config['BIOPY_CAPTURE_FOLDER'] = BIOPY_CAPTURE_FOLDER
os.makedirs(BIOPY_CAPTURE_FOLDER, exist_ok=True) # Ensure this folder exists

db = SQLAlchemy(app)

# >>>>>>>>>>>>> IMPORTANT FIX: ADDING db.create_all() HERE <<<<<<<<<<<<<
# This ensures tables are created when the app starts, both locally and on Render
with app.app_context():
    db.create_all()
    print("DEBUG: Database tables checked/created.")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# --- Database Model: User ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    patients = db.relationship('Patient', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

# --- Database Model: Patient ---
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    slides = db.relationship('Slide', backref='patient', lazy=True) # Used for both types of slides
    diagnostic_report = db.Column(db.Text, nullable=True) # Stores the final diagnostic report
    is_archived = db.Column(db.Boolean, default=False, nullable=False) # True if case is archived

    def __repr__(self):
        return f'<Patient {self.name} (User: {self.user_id})>'

# --- Database Model: Slide (Used for both general uploads and stitched images) ---
class Slide(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False) # Original filename for uploaded, or generated for stitched
    processed_filename = db.Column(db.String(255), nullable=True) # Filename of the processed image
    biopsy_coords = db.Column(db.Text, nullable=True) # Stores coordinates or description
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    def __repr__(self):
        return f'<Slide {self.filename} (Patient: {self.patient_id})>'

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Global Variables for Camera Capture and Real-Time Stitching ---
# IMPORTANT: On Render, you CANNOT access a physical webcam.
# The camera functionality will only work locally.
# For a deployed app, you'd need a different strategy (e.g., upload pre-recorded video, or use a specific camera API if available).
camera = None # Initialize camera to None
try:
    camera = cv2.VideoCapture(0) # Attempt to initialize the default webcam
    if not camera.isOpened():
        print("WARNING: Could not open default camera (index 0). Camera features will not work.")
        camera = None
except Exception as e:
    print(f"ERROR: Failed to initialize camera: {e}. Camera features will not work.")
    camera = None


GRID_ROWS, GRID_COLS = 3, 3      # Biopsy grid size (e.g., 3x3)
current_cell = [0, 0]            # [row, col] pointer, starts at top-left
cell_h, cell_w = None, None      # Will be set after the first frame is read to determine camera resolution
stitched_canvas = None           # Global composite image, updated in real-time (for preview only)
last_update_ts = 0               # Used to throttle the MJPEG stream for stitched_feed

is_capturing_continuously = False # Global flag for continuous capture
capture_thread = None # To hold the continuous capture thread

# --- Simulated Camera Settings (for Step 9) ---
current_focus = 50 # 0-100 scale
current_zoom = 50  # 0-100 scale

# --- Global variable to track captured tiles for workflow progression ---
captured_tiles_count = 0
total_tiles_needed = GRID_ROWS * GRID_COLS

# ─── HELPERS FOR REAL-TIME STITCHING & GUIDANCE ───────────────────────────────────────────
def set_cell_dims(h, w):
    """
    Calculates the dimensions of each grid cell based on frame size and grid size.
    Initializes the global stitched_canvas with black pixels.
    """
    global cell_h, cell_w, stitched_canvas
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS
    # Initialize stitched_canvas as a black image with the total dimensions of the grid
    stitched_canvas = np.zeros((cell_h * GRID_ROWS,
                                  cell_w * GRID_COLS, 3), dtype="uint8")

def draw_overlay(frame):
    """
    Draws the 3x3 grid on the live camera feed and highlights the current cell.
    Initializes cell dimensions and stitched_canvas on the first call.
    Also adds a simulated focus dot.
    """
    global cell_h, cell_w
    h, w = frame.shape[:2] # Get current frame height and width

    # Initialize cell dimensions and the stitched canvas if not already set
    if cell_h is None:
        set_cell_dims(h, w)

    # Draw vertical grid lines
    for r_line in range(1, GRID_ROWS):
        cv2.line(frame, (0, r_line * cell_h), (w, r_line * cell_h), (255, 255, 255), 1) # White lines

    # Draw horizontal grid lines
    for c_line in range(1, GRID_COLS):
        cv2.line(frame, (c_line * cell_w, 0), (c_line * cell_w, h), (255, 255, 255), 1) # White lines

    # Highlight the current focus cell with a green rectangle
    r, c = current_cell
    cv2.rectangle(frame, (c * cell_w, r * cell_h),
                  (c * cell_w + cell_w, r * cell_h + cell_h),
                  (0, 255, 0), 2) # Green rectangle, 2px thick

    # Add text guidance for the current cell
    guidance_text = f"Capture {r + 1},{c + 1}" # Display 1-indexed coordinates
    cv2.putText(frame, guidance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text

    # --- Simulated Focus Dot (Step 9) ---
    # Draw a small red circle in the center of the current cell to simulate focus
    focus_center_x = c * cell_w + cell_w // 2
    focus_center_y = r * cell_h + cell_h // 2
    cv2.circle(frame, (focus_center_x, focus_center_y), 5, (0, 0, 255), -1) # Red filled circle

    return frame


def place_on_canvas(img, r, c):
    """
    Resizes the captured image (tile) to the standard cell dimensions and
    places it onto the global stitched_canvas at the correct grid position.
    This is for the REAL-TIME PREVIEW only.
    """
    global stitched_canvas
    # Ensure image is not empty before resizing
    if img is None or img.size == 0:
        print(f"Warning: Attempted to place empty image for cell {r},{c} on real-time canvas.")
        return
    
    # Ensure cell_w and cell_h are set
    if cell_w is None or cell_h is None:
        # If not set, try to get dimensions from the current frame
        h, w = img.shape[:2]
        set_cell_dims(h, w)

    # Resize the captured image to fit exactly into one cell
    resized = cv2.resize(img, (cell_w, cell_h))
    # Place the resized image into the corresponding section of the stitched_canvas
    stitched_canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized


def gen_live_feed():
    """
    Generator function for the live camera feed (MJPEG stream).
    Continuously reads frames, draws the grid overlay, and yields JPEG bytes.
    """
    global camera, current_focus, current_zoom
    if camera is None:
        # Provide a placeholder image if camera is not available
        placeholder_text = "Camera Not Available"
        img = np.zeros((480, 640, 3), dtype="uint8") # Black image
        cv2.putText(img, placeholder_text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode(".jpg", img)
        frame_bytes = buffer.tobytes()
        while True: # Keep yielding the placeholder
            yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
                   frame_bytes + b"\r\n")
            time.sleep(1) # Don't hog CPU

    while True:
        success, frame = camera.read() # Read a frame from the camera
        if not success:
            print("Failed to read frame from camera. Is it in use by another application or not connected?")
            # Attempt to re-open camera if it was disconnected
            try:
                camera.release()
                camera = cv2.VideoCapture(0)
                time.sleep(1) # Wait a bit before retrying
                if not camera.isOpened():
                    raise IOError("Camera still not open after re-initialization attempt.")
            except Exception as e:
                print(f"ERROR: Camera re-initialization failed: {e}. Switching to placeholder.")
                # If camera fails permanently, switch to placeholder
                placeholder_text = "Camera Disconnected"
                img = np.zeros((480, 640, 3), dtype="uint8") # Black image
                cv2.putText(img, placeholder_text, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                ret, buffer = cv2.imencode(".jpg", img)
                frame_bytes = buffer.tobytes()
                while True: # Keep yielding the placeholder
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
                           frame_bytes + b"\r\n")
                    time.sleep(1) # Don't hog CPU
            continue # Skip to next iteration
        
        # Apply simulated zoom (scaling the frame)
        # Note: This is a simple simulation. Real optical zoom is more complex.
        if current_zoom != 50: # Only apply if zoom is not at default
            scale_factor = 1 + (current_zoom - 50) / 50.0 # Scale from 0.5x to 1.5x
            
            # Get original frame dimensions
            original_h, original_w = frame.shape[:2]
            
            # Calculate new dimensions
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            
            # Resize the frame
            if scale_factor > 1: # Zoom in
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Crop back to original size, centered
                start_x = int((new_w - original_w) / 2)
                start_y = int((new_h - original_h) / 2)
                frame = frame[start_y:start_y+original_h,
                              start_x:start_x+original_w]
            else: # Zoom out
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Pad back to original size, centered
                padded_frame = np.zeros((original_h, original_w, 3), dtype="uint8")
                pad_x = int((original_w - new_w) / 2)
                pad_y = int((original_h - new_h) / 2)
                padded_frame[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = frame
                frame = padded_frame


        # Apply simulated focus (blur/sharpen)
        # Note: This is a simple simulation. Real focus is more complex.
        if current_focus < 50: # Apply blur if focus is below 50
            # Map 0-49 to blur kernel size (e.g., 1 to 9, must be odd)
            blur_amount = int((50 - current_focus) / 50 * 8 + 1) # Max blur_amount = 9
            if blur_amount % 2 == 0: # Ensure kernel is odd
                blur_amount += 1
            frame = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
        elif current_focus > 50: # Simulate sharpening if focus is above 50
            # Simple sharpening using unsharp mask effect
            # Create a sharpened version and blend it
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Sharpening kernel
            sharpened = cv2.filter2D(frame, -1, kernel)
            # Blend original and sharpened based on focus level
            alpha = (current_focus - 50) / 50.0 # From 0 to 1
            frame = cv2.addWeighted(frame, 1 - alpha, sharpened, alpha, 0)


        frame = draw_overlay(frame) # Apply the grid overlay
        
        ret, buffer = cv2.imencode(".jpg", frame) # Encode frame as JPEG
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
               frame_bytes + b"\r\n") # Yield JPEG bytes for streaming


def gen_stitched_feed():
    """
    Generator function for the real-time stitched composite image (MJPEG stream).
    Continuously yields the current state of the global stitched_canvas.
    Includes a throttle to limit FPS and save bandwidth.
    """
    global last_update_ts, stitched_canvas
    while True:
        if stitched_canvas is None:
            # If canvas is not initialized yet, wait a bit and continue
            # Or provide a blank/placeholder image
            placeholder_text = "Stitched View Not Ready"
            img = np.zeros((360, 480, 3), dtype="uint8") # Black image
            cv2.putText(img, placeholder_text, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode(".jpg", img)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
                   frame_bytes + b"\r\n")
            time.sleep(1) # Don't hog CPU while waiting for canvas
            continue

        # Throttle the stream to approximately 10 FPS
        if time.time() - last_update_ts < 0.1: # 0.1 seconds = 100ms delay
            time.sleep(0.05)
            continue
        last_update_ts = time.time() # Update timestamp for throttling

        ret, buffer = cv2.imencode(".jpg", stitched_canvas) # Encode stitched canvas as JPEG
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\r\n" +
               frame_bytes + b"\r\n") # Yield JPEG bytes for streaming

# --- Continuous Capture Thread Function (Step 5 - Simplified) ---
def continuous_capture_worker():
    """
    Worker function to continuously capture tiles at intervals.
    This simulates video capture by repeatedly calling capture_biopsy_tile_internal.
    """
    global is_capturing_continuously, captured_tiles_count, total_tiles_needed
    # Ensure camera is available for continuous capture
    if camera is None:
        print("Continuous capture worker cannot start: Camera not initialized.")
        is_capturing_continuously = False # Ensure flag is false
        with app.test_request_context():
            flash("Camera not available for continuous capture. Please check camera connection.", 'error')
        return

    while is_capturing_continuously:
        with app.app_context(): # Need app context for session access
            if captured_tiles_count < total_tiles_needed:
                # Simulate a manual capture tile action
                if capture_biopsy_tile_internal():
                    captured_tiles_count += 1
                    # Simulate a "missed frame" occasionally for demonstration (Step 7)
                    if np.random.rand() < 0.02: # 2% chance to "miss" a frame
                        # Use app.test_request_context for flashing outside a direct request
                        with app.test_request_context():
                            flash(f"Missed frame detected near ({current_cell[0]+1},{current_cell[1]+1})! Please adjust camera.", 'warning')
                        print("SIMULATED: Missed frame detected!")
                else:
                    print("Continuous capture worker failed to capture tile.")
            else:
                # All tiles captured, stop continuous capture
                is_capturing_continuously = False
                with app.test_request_context(): # Use test context for flashing outside a request
                    flash("All biopsy sections captured! Proceed to final review.", 'success')
                print("All tiles captured. Stopping continuous capture.")

        time.sleep(2) # Capture a tile every 2 seconds (simulated video frame rate)

def capture_biopsy_tile_internal():
    """
    Internal function to capture a single tile, without Flask request/response.
    Used by the continuous capture worker or manual capture route.
    Saves the individual tile and updates the real-time stitched canvas.
    """
    global current_cell, stitched_canvas, camera, captured_tiles_count

    if camera is None:
        print("Camera not available for internal tile capture.")
        return False

    # Check if we've already captured all tiles
    if current_cell[0] >= GRID_ROWS or (current_cell[0] == GRID_ROWS - 1 and current_cell[1] >= GRID_COLS):
        print("Attempted to capture tile beyond grid boundaries. Capture already complete.")
        return False # Indicate no more tiles to capture

    r, c = current_cell
    ok, frame = camera.read()
    if not ok:
        print("Camera error: Could not read frame during internal capture. Attempting to re-open.")
        try:
            camera.release() # Release potentially stuck camera
            camera = cv2.VideoCapture(0) # Try to re-initialize
            time.sleep(1) # Give camera time to initialize
            ok, frame = camera.read() # Try reading again
            if not ok:
                raise IOError("Camera still not accessible after re-initialization.")
        except Exception as e:
            print(f"ERROR: Camera re-initialization failed during tile capture: {e}")
            return False # Indicate failure

    patient_id = session.get('current_patient_id')
    if patient_id:
        # Generate a unique filename for the individual tile
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # Include milliseconds
        fname = f"patient_{patient_id}_tile_{r}_{c}_{timestamp}.jpg"
        path = os.path.join(app.config['BIOPY_CAPTURE_FOLDER'], fname)
        cv2.imwrite(path, frame)
        print(f"Captured tile {r},{c} for patient {patient_id} and saved to {path}")

        # Store the path in the session for final stitching
        if 'current_patient_captured_tiles' not in session:
            session['current_patient_captured_tiles'] = []
        # Ensure we don't add duplicate paths if somehow called multiple times for same cell
        if path not in session['current_patient_captured_tiles']:
            session['current_patient_captured_tiles'].append(path)
            session.modified = True # Mark session as modified
            print(f"Added tile path to session: {path}")
    else:
        print("No patient ID in session, skipping saving tile to disk.")

    # Update the real-time stitched canvas (simple placement)
    place_on_canvas(frame, r, c)

    # Advance the grid pointer for the *next* capture
    if c < GRID_COLS - 1:
        current_cell[1] += 1
    else: # c is GRID_COLS - 1
        current_cell[1] = 0
        current_cell[0] += 1
    return True # Indicate success

def perform_advanced_stitching(patient_id):
    """
    Performs advanced image stitching using OpenCV for the final composite image.
    Retrieves individual tile paths from the session.
    Returns the stitched image (NumPy array) or None if stitching fails.
    """
    if 'current_patient_captured_tiles' not in session:
        print("No captured tiles found in session for advanced stitching.")
        return None

    # Filter tiles for the current patient (though session should already be scoped)
    # This ensures we only stitch tiles relevant to the current patient session
    tile_paths = [
        path for path in session['current_patient_captured_tiles']
        if f"patient_{patient_id}" in os.path.basename(path)
    ]

    if not tile_paths:
        print("No tile paths available for stitching.")
        return None

    # Sort tile paths to ensure correct order for stitching (important for grid layouts)
    # This is a simple sort, more robust sorting might be needed for complex patterns
    tile_paths.sort() 

    images = []
    for path in tile_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not read image from path: {path}")

    if not images:
        print("No valid images loaded for stitching.")
        return None

    # Create a Stitcher object
    # cv2.Stitcher_SCANS is often good for images captured in a grid/scan pattern
    # cv2.Stitcher_PANORAMA is for more general panoramic stitching
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS) # Or cv2.Stitcher_PANORAMA

    # Perform stitching
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Advanced stitching successful!")
        return stitched_image
    else:
        print(f"Advanced stitching failed with status: {status}")
        # Provide more specific error messages based on status
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Stitching error: Need more images or insufficient overlap.")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Stitching error: Homography estimation failed (not enough matching features).")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Stitching error: Camera parameters adjustment failed.")
        return None


# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html') # Assuming index.html is your landing page

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if 'username' in session:
        print(f"DEBUG: Login page - User already in session: {session.get('username')}. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        print(f"DEBUG: Login attempt for email: {email}") # Debug print
        user = User.query.filter_by(email=email).first()
        
        if user:
            print(f"DEBUG: User found: {user.email}") # Debug print
            if user.check_password(password):
                session['username'] = user.email
                session['user_id'] = user.id
                session.modified = True # Explicitly mark session as modified
                print(f"DEBUG: Login successful for {user.email}. Session after setting: {session}") # Debug print
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password.', 'error')
                print(f"DEBUG: Password mismatch for {user.email}") # Debug print
                return render_template('login.html')
        else:
            flash('Invalid email or password.', 'error')
            print(f"DEBUG: User not found for email: {email}") # Debug print
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if 'username' in session:
        print(f"DEBUG: Register page - User already in session: {session.get('username')}. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('An account with this email already exists. Please login.', 'error')
            return render_template('register.html')
        new_user = User(email=email)
        new_user.set_password(password)
        try:
            db.session.add(new_user)
            db.session.commit()
            print(f"DEBUG: New user registered: {new_user.email}") # Debug print
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login_page'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred during registration. Please try again. ({e})', 'error')
            print(f"DEBUG: Error during registration: {e}") # Debug print
            return render_template('register.html')
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    print(f"DEBUG: Dashboard route accessed. Current session: {session}") # Debug print at start of dashboard route
    if 'username' not in session:
        flash('Please login to access the dashboard.', 'error')
        print("DEBUG: User not in session. Redirecting to login page.")
        return redirect(url_for('login_page'))

    # Clear any old processed image/coords from session, as they are no longer displayed on dashboard
    session.pop('processed_image_url', None)
    session.pop('biopsy_coordinates', None)

    user_id = session.get('user_id')
    current_patient_id = session.get('current_patient_id')
    current_patient = None
    if current_patient_id:
        current_patient = Patient.query.get(current_patient_id)
        if not current_patient: # Patient not found, clear session to restart workflow
            session.pop('current_patient_id', None)
            session.pop('current_patient_name', None)
            session.pop('slide_uploaded_for_current_patient_flag', None)
            session.pop('biopsy_region_selected_flag', None)
            session.pop('uploaded_slide_url', None)
            session.pop('current_patient_captured_tiles', None) # Clear captured tiles
            session.modified = True
            current_patient_name = 'N/A' # Reset for template
            patient_active = False
        else:
            current_patient_name = current_patient.name
            patient_active = True
    else:
        current_patient_name = 'N/A'
        patient_active = False

    slide_uploaded_for_current_patient = session.get('slide_uploaded_for_current_patient_flag', False)
    biopsy_region_selected = session.get('biopsy_region_selected_flag', False)
    uploaded_slide_url = session.get('uploaded_slide_url', None)
    camera_mode_active = session.get('camera_mode_active', False) # New: Get camera mode preference

    # Determine if capture is complete for workflow progression (Phase 3)
    capture_complete = (captured_tiles_count >= total_tiles_needed)

    # If capture is complete, stop continuous capture if it's still running
    global is_capturing_continuously, capture_thread
    if capture_complete and is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped as all tiles are captured.")
        # Flash message for completion is now handled by the continuous_capture_worker thread

    # --- Generate final stitched image for Phase 4 display if capture is complete ---
    final_stitched_display_url = None
    if capture_complete and patient_active:
        print("DEBUG: Generating final stitched image for dashboard display...")
        stitched_img_np = perform_advanced_stitching(current_patient_id)
        if stitched_img_np is not None:
            # Save the stitched image temporarily for display
            display_filename = f"final_stitched_display_{current_patient_id}.jpg"
            display_path = os.path.join(app.config['PROCESSED_FOLDER'], display_filename)
            cv2.imwrite(display_path, stitched_img_np)
            final_stitched_display_url = url_for('static', filename=f'processed/{display_filename}')
            print(f"DEBUG: Final stitched image saved for display at: {final_stitched_display_url}")
        else:
            print("DEBUG: Failed to generate final stitched image for display.")
            flash("Could not generate high-quality stitched image for display. Please check captured tiles.", "warning")

    return render_template('dashboard.html',
                           username=session['username'],
                           rows=GRID_ROWS,
                           cols=GRID_COLS,
                           current_patient_name=current_patient_name,
                           patient_active=patient_active,
                           slide_uploaded_for_current_patient=slide_uploaded_for_current_patient,
                           biopsy_region_selected=biopsy_region_selected,
                           uploaded_slide_url=uploaded_slide_url,
                           is_capturing_continuously=is_capturing_continuously,
                           current_focus=current_focus,
                           current_zoom=current_zoom,
                           capture_complete=capture_complete, # New flag for workflow progression
                           current_patient=current_patient, # Pass the patient object for report/archive
                           camera_mode_active=camera_mode_active, # New: Pass camera mode preference
                           final_stitched_display_url=final_stitched_display_url # New: URL for final stitched image
                           )

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    session.pop('current_patient_id', None)
    session.pop('current_patient_name', None)
    session.pop('slide_uploaded_for_current_patient_flag', None)
    session.pop('biopsy_region_selected_flag', None)
    session.pop('uploaded_slide_url', None)
    session.pop('processed_image_url', None)
    session.pop('biopsy_coordinates', None)
    session.pop('camera_mode_active', None) # New: Clear camera mode preference
    session.modified = True # Mark session as modified
    
    # Clean up individual tile images for the logged-out user (if any are left)
    user_id = session.get('user_id')
    if user_id:
        cleanup_tile_images_for_user(user_id) # Call cleanup for all patient tiles of this user
    session.pop('current_patient_captured_tiles', None) # Clear captured tiles from session
    session.modified = True

    # Stop continuous capture if active on logout
    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on logout.")

    # Reset global camera state on logout
    current_cell = [0, 0]
    stitched_canvas = None
    captured_tiles_count = 0 # Reset captured tiles count
    flash('You have been logged out.', 'success')
    print("DEBUG: User logged out. Session cleared.")
    return redirect(url_for('home'))

@app.route('/save_patient_details', methods=['POST'])
def save_patient_details():
    if request.method == 'POST':
        if 'user_id' not in session:
            flash('Please log in to save patient details.', 'error')
            return redirect(url_for('login_page'))
        try:
            patient_name = request.form['patientName']
            patient_age = int(request.form['patientAge'])
            patient_sex = request.form['patientSex']
            patient_date_str = request.form['patientDate']
            patient_date = datetime.strptime(patient_date_str, '%Y-%m-%d').date()
            user_id = session['user_id']

            new_patient = Patient(
                name=patient_name,
                age=patient_age,
                sex=patient_sex,
                date=patient_date,
                user_id=user_id
            )
            db.session.add(new_patient)
            db.session.commit()

            session['current_patient_id'] = new_patient.id
            session['current_patient_name'] = new_patient.name
            session['slide_uploaded_for_current_patient_flag'] = False # Force slide upload next
            session['biopsy_region_selected_flag'] = False # Force region selection next
            session.pop('uploaded_slide_url', None) # Clear previous slide URL
            session['camera_mode_active'] = False # Default to file upload mode for new patient
            session['current_patient_captured_tiles'] = [] # Initialize list for captured tile paths
            session.modified = True # Explicitly mark session as modified

            # Reset capture state for new patient
            global current_cell, stitched_canvas, captured_tiles_count
            current_cell = [0, 0]
            stitched_canvas = None
            captured_tiles_count = 0

            flash('Patient details saved successfully! Now, please upload the biopsy slide image.', 'success')
            print(f"DEBUG: Patient details saved. Session: {session}")
            return redirect(url_for('dashboard'))

        except ValueError:
            flash('Invalid age or date format. Please check your input.', 'error')
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while saving patient details: {e}', 'error')
            print(f"DEBUG: Error saving patient details: {e}")
    return redirect(url_for('dashboard'))

@app.route('/clear_patient_session')
def clear_patient_session():
    """
    Clears the current patient and slide/region session, forcing the dashboard to show
    the patient details input form again. Also cleans up captured tile images.
    """
    patient_id_to_clear = session.get('current_patient_id') # Get ID before popping
    
    session.pop('current_patient_id', None)
    session.pop('current_patient_name', None)
    session.pop('slide_uploaded_for_current_patient_flag', None)
    session.pop('biopsy_region_selected_flag', None)
    session.pop('uploaded_slide_url', None)
    session.pop('camera_mode_active', None) # New: Clear camera mode preference
    session.modified = True
    
    # Clean up individual tile images for the cleared patient
    if patient_id_to_clear:
        cleanup_tile_images_for_patient(patient_id_to_clear)
    session.pop('current_patient_captured_tiles', None) # Clear captured tiles from session
    session.modified = True # Explicitly mark session as modified

    # Stop continuous capture if active when clearing patient session
    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on clear patient session.")

    # Reset global camera state
    current_cell = [0, 0]
    stitched_canvas = None
    captured_tiles_count = 0 # Reset captured tiles count

    # Re-initialize stitched_canvas with black pixels based on typical camera resolution
    # Only if camera is available
    if camera and camera.isOpened():
        success, frame_for_dims = camera.read()
        if success:
            set_cell_dims(frame_for_dims.shape[0], frame_for_dims.shape[1])
            flash('Biopsy capture session reset. You can start capturing new tiles.', 'info')
        else:
            flash('Biopsy capture session reset, but camera could not be accessed to re-initialize canvas. Please ensure camera is connected.', 'warning')
            print("DEBUG: Warning: Camera not accessible during reset_biopsy_capture_session to re-initialize stitched_canvas.")
    else:
        flash('Biopsy capture session reset. Camera is not available.', 'info')
        print("DEBUG: Camera not available, skipping stitched_canvas re-initialization during reset.")


    print(f"DEBUG: Patient session cleared. Session: {session}")
    return redirect(url_for('dashboard'))


def cleanup_tile_images_for_user(user_id):
    """
    Deletes all individual tile images associated with any patient of a specific user ID.
    This is a broader cleanup, useful on logout.
    """
    # First, get all patient IDs for this user
    # Need app context for database query
    with app.app_context():
        patient_ids = [p.id for p in Patient.query.filter_by(user_id=user_id).all()]
    total_deleted = 0
    for p_id in patient_ids:
        search_pattern = os.path.join(app.config['BIOPY_CAPTURE_FOLDER'], f"patient_{p_id}_tile_*.jpg")
        files_to_delete = glob.glob(search_pattern)
        for f in files_to_delete:
            try:
                os.remove(f)
                total_deleted += 1
            except OSError as e:
                print(f"Error deleting file {f} for user {user_id}: {e}")
    print(f"Cleaned up {total_deleted} individual tile images for user {user_id} across all their patients.")


# --- ROUTE: Handle Traditional Slide Upload and Analysis ---
@app.route('/upload_slide', methods=['POST'])
def upload_slide():
    """
    Handles traditional slide image uploads. After upload, redirects to dashboard
    to allow user to select region on the uploaded image.
    """
    if 'user_id' not in session:
        flash('Please log in to upload slides.', 'error')
        return redirect(url_for('login_page'))

    if not session.get('current_patient_id'):
        flash('Please save patient details first before uploading a slide.', 'error')
        return redirect(url_for('dashboard'))

    if 'slideImage' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('dashboard'))

    file = request.files['slideImage']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['uploaded_slide_url'] = url_for('static', filename=f'uploads/{filename}')
        session['slide_uploaded_for_current_patient_flag'] = True
        session['biopsy_region_selected_flag'] = False # Reset, forcing region selection next
        session['camera_mode_active'] = False # Ensure we're in file upload mode if this was used
        session['current_patient_captured_tiles'] = [] # Initialize list for captured tile paths
        session.modified = True # Explicitly mark session as modified

        flash('Slide uploaded successfully! Now, please select the region of interest on the slide.', 'success')
        print(f"DEBUG: Slide uploaded. Session: {session}")
        return redirect(url_for('dashboard'))

    else:
        flash('Allowed image types are png, jpg, jpeg, gif', 'error')
        return redirect(url_for('dashboard'))

@app.route('/upload_camera_image', methods=['POST'])
def upload_camera_image():
    """
    Handles image data sent from the frontend camera capture.
    Decodes base64, saves the image, and updates session flags.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    if not session.get('current_patient_id'):
        return jsonify(status='error', message='Please save patient details first.'), 400

    data = request.json
    image_data_url = data.get('imageData')

    if not image_data_url:
        return jsonify(status='error', message='No image data provided.'), 400

    try:
        # Extract base64 part (e.g., "data:image/jpeg;base64,..." -> "...")
        header, encoded = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded)

        # Determine file extension from header
        mime_type = header.split(':')[1].split(';')[0]
        if 'jpeg' in mime_type:
            ext = 'jpg'
        elif 'png' in mime_type:
            ext = 'png'
        else:
            return jsonify(status='error', message='Unsupported image format.'), 400

        filename = f"camera_capture_{session['current_patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(image_bytes)

        session['uploaded_slide_url'] = url_for('static', filename=f'uploads/{filename}')
        session['slide_uploaded_for_current_patient_flag'] = True
        session['biopsy_region_selected_flag'] = False # Reset, forcing region selection next
        session['camera_mode_active'] = True # Ensure we stay in camera mode if this was used
        session['current_patient_captured_tiles'] = [] # Initialize list for captured tile paths
        session.modified = True # Explicitly mark session as modified

        print(f"DEBUG: Camera image uploaded. Session: {session}")
        return jsonify(status='success', message='Camera image uploaded successfully.')

    except Exception as e:
        print(f"DEBUG: Error processing camera image upload: {e}")
        return jsonify(status='error', message=f'Failed to process image: {e}'), 500


@app.route('/set_camera_mode_active', methods=['POST'])
def set_camera_mode_active():
    """
    Sets the session variable to remember if the camera mode was active.
    This helps the dashboard load in the correct tab.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    
    data = request.json
    is_active = data.get('active', False) # Default to False if not provided

    session['camera_mode_active'] = is_active
    session.modified = True # Explicitly mark session as modified
    print(f"DEBUG: Camera mode set to {is_active}. Session: {session}")
    return jsonify(status='success', camera_mode_active=is_active)


@app.route('/confirm_biopsy_region', methods=['POST'])
def confirm_biopsy_region():
    """
    Receives the selected region coordinates from the frontend and marks
    the biopsy region as selected for the current patient/slide.
    """
    if 'user_id' not in session:
        return ('Access Denied', 403)
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag')):
        return ('Workflow step incomplete. Please upload a slide first.', 400)

    data = request.json
    selected_coords = {
        'x': data.get('x'),
        'y': data.get('y'),
        'width': data.get('width'),
        'height': data.get('height')
    }
    session['biopsy_region_coords'] = selected_coords
    session['biopsy_region_selected_flag'] = True # Mark region as selected
    session.modified = True # Explicitly mark session as modified

    flash('Biopsy region selected successfully! You can now proceed with live biopsy capture.', 'success')
    print(f"DEBUG: Biopsy region confirmed. Session: {session}")
    return ('', 204) # 204 No Content, indicating success


# --- NEW ROUTES FOR REAL-TIME CAMERA CAPTURE AND STITCHING ---

@app.route('/live_video_feed')
def live_video_feed():
    """
    Streams the live camera feed with grid overlay to the web page.
    Requires user to be logged in, patient active, slide uploaded, and region selected.
    """
    if 'user_id' not in session:
        print("DEBUG: Live feed access denied - no user in session.")
        return Response("Access Denied", status=403)
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        print("DEBUG: Live feed access denied - workflow incomplete.")
        return Response("Camera feed not available. Please complete previous workflow steps.", status=403)

    return Response(gen_live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stitched_biopsy_feed')
def stitched_biopsy_feed():
    """
    Streams the real-time stitched composite image to the web page.
    Requires user to be logged in, patient active, slide uploaded, and region selected.
    """
    if 'user_id' not in session:
        print("DEBUG: Stitched feed access denied - no user in session.")
        return Response("Access Denied", status=403)
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        print("DEBUG: Stitched feed access denied - workflow incomplete.")
        return Response("Stitched feed not available. Please complete previous workflow steps.", status=403)

    return Response(gen_stitched_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/capture_biopsy_tile", methods=["POST"])
def capture_biopsy_tile_route(): # Renamed to avoid conflict with internal function
    """
    Captures a single frame from the camera, saves it, updates the stitched canvas,
    and advances the current_cell pointer to the next grid position.
    This route is called manually by the frontend.
    """
    if 'user_id' not in session:
        print("DEBUG: Capture tile denied - no user in session.")
        return ('Access Denied', 403)
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        print("DEBUG: Capture tile denied - workflow incomplete.")
        return ('Workflow step incomplete. Please complete previous workflow steps.', 400)

    global captured_tiles_count, total_tiles_needed
    if captured_tiles_count < total_tiles_needed:
        if capture_biopsy_tile_internal():
            captured_tiles_count += 1
            if captured_tiles_count >= total_tiles_needed:
                # All tiles captured, stop continuous capture if it's running
                global is_capturing_continuously, capture_thread
                if is_capturing_continuously:
                    is_capturing_continuously = False
                    if capture_thread and capture_thread.is_alive():
                        capture_thread.join(timeout=5)
                        print("DEBUG: Continuous capture thread stopped as all tiles are captured.")
                flash("All biopsy sections captured! You can now review and generate the report.", 'success')
            print(f"DEBUG: Tile captured. Current captured count: {captured_tiles_count}. Session: {session}")
            return ("", 204) # 204 No Content, indicating successful request without new content
        else:
            print("DEBUG: Failed to capture tile internally.")
            return ("Failed to capture tile.", 500)
    else:
        flash("All tiles already captured for this session. Proceed to final review or reset.", 'info')
        print("DEBUG: All tiles already captured.")
        return ("All tiles already captured.", 200)


@app.route('/start_continuous_capture', methods=['POST'])
def start_continuous_capture():
    """
    Starts the continuous capture thread.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        return jsonify(status='error', message='Workflow step incomplete. Please complete previous workflow steps.'), 400

    global is_capturing_continuously, capture_thread, captured_tiles_count, total_tiles_needed
    
    if captured_tiles_count >= total_tiles_needed:
        flash("All tiles already captured for this session. Please reset if you want to capture again.", 'info')
        return jsonify(status='capture_complete')

    if not is_capturing_continuously:
        is_capturing_continuously = True
        capture_thread = threading.Thread(target=continuous_capture_worker)
        capture_thread.daemon = True # Allow main program to exit even if thread is running
        capture_thread.start()
        flash('Continuous video capture started.', 'success')
        print("DEBUG: Continuous capture started.")
        return jsonify(status='started')
    print("DEBUG: Continuous capture already started.")
    return jsonify(status='already_started')

@app.route('/stop_continuous_capture', methods=['POST'])
def stop_continuous_capture():
    """
    Stops the continuous capture thread.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    global is_capturing_continuously, capture_thread
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5) # Wait for thread to finish
            print("DEBUG: Continuous capture thread stopped via API.")
        flash('Continuous video capture stopped.', 'info')
        return jsonify(status='stopped')
    print("DEBUG: Continuous capture already stopped.")
    return jsonify(status='already_stopped')

@app.route('/get_camera_guidance')
def get_camera_guidance():
    """
    Provides real-time directional guidance based on the current cell.
    Also provides current focus/zoom and capture completion status.
    """
    r, c = current_cell
    next_row = r
    next_col = c + 1

    guidance_message = ""
    # Check if all tiles are captured
    if captured_tiles_count >= total_tiles_needed:
        guidance_message = "All sections captured! Proceed to final review and report generation."
    elif next_col >= GRID_COLS: # End of current row
        next_row += 1
        next_col = 0
        if next_row >= GRID_ROWS: # This case should ideally be caught by captured_tiles_count >= total_tiles_needed
            guidance_message = "All sections captured! Proceed to final review and report generation."
        else:
            guidance_message = f"Move to next row: Row {next_row + 1}, Column {next_col + 1} (move camera down and reset left)."
    else:
        guidance_message = f"Next section: Row {next_row + 1}, Column {next_col + 1} (move camera right)."

    return jsonify({
        'guidance': guidance_message,
        'current_cell': {'row': r, 'col': c},
        'next_cell': {'row': next_row, 'col': next_col},
        'focus_level': current_focus,
        'zoom_level': current_zoom,
        'capture_complete': (captured_tiles_count >= total_tiles_needed) # Send capture status
    })

@app.route('/set_camera_focus', methods=['POST'])
def set_camera_focus():
    """
    Sets the simulated camera focus level.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    data = request.json
    focus_level = data.get('focus_level')
    if focus_level is not None and 0 <= focus_level <= 100:
        global current_focus
        current_focus = int(focus_level)
        print(f"DEBUG: Focus set to {current_focus}")
        return jsonify(status='success', focus_level=current_focus)
    return jsonify(status='error', message='Invalid focus level'), 400

@app.route('/set_camera_zoom', methods=['POST'])
def set_camera_zoom():
    """
    Sets the simulated camera zoom level.
    """
    if 'user_id' not in session:
        return jsonify(status='error', message='Authentication missing.'), 403
    data = request.json
    zoom_level = data.get('zoom_level')
    if zoom_level is not None and 0 <= zoom_level <= 100:
        global current_zoom
        current_zoom = int(zoom_level)
        print(f"DEBUG: Zoom set to {current_zoom}")
        return jsonify(status='success', zoom_level=current_zoom)
    return jsonify(status='error', message='Invalid zoom level'), 400


@app.route("/reset_biopsy_capture_session")
def reset_biopsy_capture_session():
    """
    Resets the camera capture state (current_cell, stitched_canvas, captured_tiles_count)
    for the current patient. Also cleans up captured tile images.
    """
    if 'user_id' not in session:
        flash('Please log in to reset.', 'error')
        return redirect(url_for('login_page'))
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        flash('Cannot reset biopsy capture. Please ensure a patient is selected, a slide is uploaded, and a region is selected.', 'error')
        return redirect(url_for('dashboard'))

    # Stop continuous capture if active
    global is_capturing_continuously, capture_thread, current_cell, stitched_canvas, captured_tiles_count
    if is_capturing_continuously:
        is_capturing_continuously = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=5)
            print("DEBUG: Continuous capture thread stopped on reset.")

    # Clean up individual tile images for the current patient
    patient_id = session.get('current_patient_id')
    if patient_id:
        cleanup_tile_images_for_patient(patient_id)
    session.pop('current_patient_captured_tiles', None) # Clear captured tiles from session
    session.modified = True # Explicitly mark session as modified

    current_cell = [0, 0] # Reset grid pointer
    stitched_canvas = None # Clear real-time stitched canvas
    captured_tiles_count = 0 # Reset captured tiles count

    # Re-initialize stitched_canvas with black pixels based on typical camera resolution
    # Only if camera is available
    if camera and camera.isOpened():
        success, frame_for_dims = camera.read()
        if success:
            set_cell_dims(frame_for_dims.shape[0], frame_for_dims.shape[1])
            flash('Biopsy capture session reset. You can start capturing new tiles.', 'info')
        else:
            flash('Biopsy capture session reset, but camera could not be accessed to re-initialize canvas. Please ensure camera is connected.', 'warning')
            print("DEBUG: Warning: Camera not accessible during reset_biopsy_capture_session to re-initialize stitched_canvas.")
    else:
        flash('Biopsy capture session reset. Camera is not available.', 'info')
        print("DEBUG: Camera not available, skipping stitched_canvas re-initialization during reset.")


    print(f"DEBUG: Biopsy capture session reset. Session: {session}")
    return redirect(url_for('dashboard'))


@app.route("/download_stitched_biopsy")
def download_stitched_biopsy():
    """
    Allows downloading the current stitched composite image.
    Also saves the stitched image as a Slide entry in the database.
    Now uses advanced stitching for the final image.
    """
    if 'user_id' not in session:
        flash('Access Denied. Please log in.', 'error')
        return redirect(url_for('login_page'))
    if not (session.get('current_patient_id') and
            session.get('slide_uploaded_for_current_patient_flag') and
            session.get('biopsy_region_selected_flag')):
        flash('Workflow step incomplete. Please complete previous workflow steps before downloading.', 'error')
        return redirect(url_for('dashboard'))

    patient_id = session['current_patient_id']
    
    # Perform advanced stitching for the download
    stitched_img_np = perform_advanced_stitching(patient_id)

    if stitched_img_np is None:
        flash("Failed to generate high-quality stitched image for download. Please ensure enough tiles were captured and have sufficient overlap.", 'error')
        return redirect(url_for('dashboard'))
    
    # Save the stitched image to a temporary file for download
    stitched_filename = f"patient_{patient_id}_seamless_biopsy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    stitched_path = os.path.join(app.config['PROCESSED_FOLDER'], stitched_filename) # Save to processed folder
    
    try:
        cv2.imwrite(stitched_path, stitched_img_np)
        print(f"DEBUG: Seamless stitched image saved to: {stitched_path}")

        # Retrieve the selected biopsy region coordinates from the session
        selected_region = session.get('biopsy_region_coords', {})
        biopsy_coords_str = f"Stitched from {GRID_ROWS}x{GRID_COLS} grid captures. Selected region on original slide: {selected_region}"

        # Save the stitched image as a new "slide" entry in the database
        new_slide = Slide(
            filename=stitched_filename, # Use the stitched filename
            processed_filename=stitched_filename, # It's already processed
            biopsy_coords=biopsy_coords_str, # Store the selected region info
            patient_id=patient_id
        )
        db.session.add(new_slide)
        db.session.commit()

        flash('Seamless stitched biopsy image downloaded and saved to patient records!', 'success')
        return send_file(stitched_path, as_attachment=True, download_name=stitched_filename)
    except Exception as e:
        db.session.rollback()
        flash(f'Error downloading or saving stitched image: {e}', 'error')
        print(f"DEBUG: Error in download_stitched_biopsy: {e}")
        return redirect(url_for('dashboard'))


# --- NEW ROUTES FOR PHASE 3: REPORTING AND ARCHIVING ---
@app.route('/save_diagnostic_report', methods=['POST'])
def save_diagnostic_report():
    """
    Saves the diagnostic report text for the current patient. (Step 12)
    """
    if 'user_id' not in session or not session.get('current_patient_id'):
        return jsonify(status='error', message='Authentication or patient session missing.'), 403
    
    patient_id = session['current_patient_id']
    report_text = request.json.get('report_text')

    patient = Patient.query.get(patient_id)
    if patient:
        patient.diagnostic_report = report_text
        try:
            db.session.commit()
            flash('Diagnostic report saved successfully!', 'success')
            print(f"DEBUG: Diagnostic report saved for patient {patient_id}.")
            return jsonify(status='success')
        except Exception as e:
            db.session.rollback()
            print(f"DEBUG: Error saving diagnostic report: {e}")
            return jsonify(status='error', message=f'Failed to save report: {e}'), 500
    return jsonify(status='error', message='Patient not found.'), 404

@app.route('/archive_case', methods=['POST'])
def archive_case():
    """
    Marks the current patient's case as archived. (Step 13)
    Also cleans up individual tile images for the archived patient.
    """
    if 'user_id' not in session or not session.get('current_patient_id'):
        return jsonify(status='error', message='Authentication or patient session missing.'), 403

    patient_id = session['current_patient_id']
    patient = Patient.query.get(patient_id)
    if patient:
        patient.is_archived = True
        try:
            db.session.commit()
            # Clear patient session after archiving to start a new case
            session.pop('current_patient_id', None)
            session.pop('current_patient_name', None)
            session.pop('slide_uploaded_for_current_patient_flag', None)
            session.pop('biopsy_region_selected_flag', None)
            session.pop('uploaded_slide_url', None)
            session.pop('camera_mode_active', None) # New: Clear camera mode preference
            session.modified = True
            
            # Clean up individual tile images for the archived patient
            cleanup_tile_images_for_patient(patient_id)
            session.pop('current_patient_captured_tiles', None) # Clear captured tiles from session
            session.modified = True # Explicitly mark session as modified

            # Also reset global camera state
            global current_cell, stitched_canvas, captured_tiles_count
            current_cell = [0, 0]
            stitched_canvas = None
            captured_tiles_count = 0

            flash('Case archived successfully! You can now start a new patient workflow.', 'success')
            print(f"DEBUG: Case archived for patient {patient_id}. Session: {session}")
            return jsonify(status='success')
        except Exception as e:
            db.session.rollback()
            print(f"DEBUG: Error archiving case: {e}")
            return jsonify(status='error', message=f'Failed to archive case: {e}'), 500
    return jsonify(status='error', message='Patient not found.'), 404


# --- Run the Application ---
# This block is for local development only. Gunicorn will import 'app' directly.
if __name__ == '__main__':
    # db.create_all() is now handled above, outside this block, for Render compatibility.
    app.run(debug=True)


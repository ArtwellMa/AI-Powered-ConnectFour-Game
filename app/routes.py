import os
import random # Keep random for potential future use or fallbacks
import logging
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from app.models import User
from app import db, bcrypt
# Removed get_ai_opponent import here, as AI logic is handled by game_routes

logger = logging.getLogger(__name__)

main_routes = Blueprint('main', __name__)
auth_routes = Blueprint('auth', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_routes.route('/')
def index():
    # Redirect authenticated users directly to the game
    if current_user.is_authenticated:
        return redirect(url_for('main.game'))
    return render_template('index.html') # Or potentially a landing page

@main_routes.route('/game')
@login_required
def game():
    # This route simply serves the game page template
    return render_template('game.html')

# *** REMOVED /get_ai_move endpoint - Moved to game_routes.py ***

# *** REMOVED /test_deepseek endpoint - Moved to game_routes.py ***


@auth_routes.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.game'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            # Redirect to the game page after successful login
            next_page = request.args.get('next') # Handle redirects if user tried accessing protected page first
            return redirect(next_page or url_for('main.game'))
        else:
            flash('Invalid username or password', 'error')
            # Return the login page again on failure
            return render_template('login.html'), 401 # Indicate unauthorized attempt

    # For GET requests, just render the login page
    return render_template('login.html')

@auth_routes.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.game'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        profile_picture = request.files.get('profile_picture')

        # --- Input Validation ---
        if not username or not email or not password or not confirm_password:
             flash('All fields except profile picture are required.', 'error')
             return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('auth.register'))

        if len(password) < 6: # Example minimum password length
            flash('Password must be at least 6 characters long.', 'error')
            return redirect(url_for('auth.register'))

        # --- Check Existing Users ---
        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.', 'error')
            return redirect(url_for('auth.register'))
        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'error')
            return redirect(url_for('auth.register'))

        # --- Process Profile Picture ---
        filename = 'default.jpg' # Default picture
        if profile_picture and allowed_file(profile_picture.filename):
            try:
                # Ensure the upload directory exists
                upload_folder = os.path.join(current_app.root_path, 'static', 'profile_pics')
                os.makedirs(upload_folder, exist_ok=True)

                # Create a secure filename based on user ID potentially, or just secure the original
                filename = secure_filename(profile_picture.filename)
                # To avoid collisions, could prefix with user ID or a timestamp after user creation
                # For now, just using the secured name
                filepath = os.path.join(upload_folder, filename)
                profile_picture.save(filepath)
                filename = f'profile_pics/{filename}' # Store relative path for url_for
            except Exception as e:
                logger.error(f"Error saving profile picture: {e}")
                flash('Error uploading profile picture. Using default.', 'warning')
                filename = 'default.jpg' # Revert to default on error

        # --- Create New User ---
        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(
                username=username,
                email=email,
                password_hash=hashed_password,
                profile_picture=filename # Use the potentially updated filename
            )
            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))

        except Exception as e:
            db.session.rollback() # Rollback transaction on error
            logger.error(f"Error creating user: {e}")
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('auth.register'))

    # For GET requests, render the registration page
    return render_template('register.html')

@auth_routes.route('/logout')
@login_required # Ensure only logged-in users can log out
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index')) # Redirect to the main index/landing page

# Optional: Add a profile view/edit route if needed later
# @main_routes.route('/profile')
# @login_required
# def profile():
#     return render_template('profile.html', user=current_user)
@main_routes.route('/replay/<game_id>')
@login_required
def replay(game_id):
    return render_template('replay.html', game_id=game_id)
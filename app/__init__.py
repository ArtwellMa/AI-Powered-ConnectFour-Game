from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)  # Initialize migrate here

    from app.routes import main_routes, auth_routes
    app.register_blueprint(main_routes)
    app.register_blueprint(auth_routes)

    # Import and register blueprints
    from app.game_routes import game_bp
    app.register_blueprint(game_bp, url_prefix='/api')
    
    
    # Import models after app creation to avoid circular imports
    from app import models

    return app

from app import models
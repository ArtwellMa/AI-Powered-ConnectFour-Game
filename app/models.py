from app import db, login_manager, bcrypt
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    profile_picture = db.Column(db.String(120), nullable=False, default='default.jpg')
    ai_metrics = db.relationship('AIMetric', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class AIMetric(db.Model):
    __tablename__ = 'ai_metric'  # Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    algorithm = db.Column(db.String(20), nullable=False)
    move_time = db.Column(db.Float, nullable=False)
    move_quality = db.Column(db.Float, nullable=False)
    difficulty = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def __repr__(self):
        return f"<AIMetric {self.algorithm} {self.timestamp}>"

    def to_dict(self):
        return {
            'id': self.id,
            'algorithm': self.algorithm,
            'move_time': self.move_time,
            'move_quality': self.move_quality,
            'difficulty': self.difficulty,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id
        }
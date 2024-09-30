from datetime import datetime
from application import db

class survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime,default=datetime.utcnow, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    ever_married = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    graduated = db.Column(db.String(10), nullable=False)
    profession = db.Column(db.String(50), nullable=False)
    work_experience = db.Column(db.Integer, nullable=False)
    spending_score = db.Column(db.String(20), nullable=False)
    family_size = db.Column(db.Integer, nullable=False)
    cluster = db.Column(db.String(1), nullable=False)

def __str__(self):
    return self.id
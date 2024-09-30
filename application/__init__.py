from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///surveryDB.db'
app.config['SECRET_KEY'] = 'This is a secret'

db = SQLAlchemy(app)
app.app_context().push()

from application import routes
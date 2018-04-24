# -*- coding=utf-8 -*-
# author: paddyguan

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import config

db = SQLAlchemy()


def register_blueprints(app):
    from apps.main import bp_index
    app.register_blueprint(bp_index.bp)


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    db.init_app(app)

    register_blueprints(app)

    return app


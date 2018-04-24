# -*- coding=utf-8 -*-
# author: paddyguan

from apps import db

class Pic_girl(db.Model):
    __tablename__ = 'pic_girl'

    id = db.Column(db.Integer, primary_key=True)
    dimension = db.Column(db.String(50))
    pic_md5 = db.Column(db.String(50))
    pic = db.Column(db.LargeBinary)

    def __init__(self):
        pass

    def __repr__(self):
        pass
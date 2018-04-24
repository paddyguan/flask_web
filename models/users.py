# -*- coding=utf-8 -*-
# author: paddyguan

from apps import db


class User(db.Model):
    __tablename__ = 'users'

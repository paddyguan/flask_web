# -*- coding=utf-8 -*-
# author: paddyguan

from flask import Blueprint
from apps import db

bp = Blueprint('bp_index', __name__)

@bp.route('/get_pic?<type>', methods=['GET'])
def get_pic(type):
    data = 0
    db.session.query('')


    return data

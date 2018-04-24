# -*- coding=utf-8 -*-
# author: paddyguan

from flask import Blueprint
from flask import render_template

bp = Blueprint('bp_index', __name__)


@bp.route('/', methods=['GET'])
def index():
   return render_template('index.html')



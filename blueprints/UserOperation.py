import time
from flask import Blueprint, request
from utils.DealWithData import formatConvert
from utils.globalVar import resultPath
from predict import predictByCSV as pbc

import os, sys, random, string

bp = Blueprint('UserOperation', __name__, url_prefix='/user')
basedir = os.path.abspath(os.path.dirname(__file__))


@bp.route('/predict', methods=['POST'])
def predict():
    molecule = request.get_json()['molecule']
    filename = time.strftime('%Y-%m-%d-%H-%M-%S')
    formatConvert(molecule, filename)

    # 属性预测
    HOMO = pbc.getHOMO(resultPath + filename + "_HOMO.csv")
    LUMO = pbc.getLUMO(resultPath + filename + "_LUMO.csv")

    return {
        "code": 0,
        "data": [{
            "Name": molecule,
            "HOMO": HOMO,
            "LUMO": LUMO,
        }]
    }


@bp.route('/save-picture/', methods=['POST'])
def save_picture():
    # 图片对象
    file_obj = request.files.get('file')
    # 图片名字
    file_name = request.form.get('fileName')
    # 图片保存的路径
    save_path = os.path.abspath(os.path.dirname(__file__) + '\\') + '\\' + str(
        file_name)
    print(save_path)
    # 保存
    file_obj.save(save_path)

    moleName = request.form.get('molName')
    extraInfo = request.form.get('extraInfo')
    print(moleName)
    print(extraInfo)
    return '图片保存成功'


@bp.route('/receiveMoleInfo', methods=['POST'])
def receiveMoleInfo():
    moleName = request.form.get('molName')
    extraInfo = request.form.get('extraInfo')
    # 接收图片对象
    file_obj = request.files.get('file')
    # 图片名字
    file_name = request.form.get('fileName')

    parent_dir = os.path.dirname(basedir)
    # 生成伪随机数作为文件名
    random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    save_path = os.path.join(parent_dir, 'img', 'UserSubmitPic', random_str + str(
        file_name))
    print(save_path)
    file_obj.save(save_path)

    return {
        "code": 200,
    }

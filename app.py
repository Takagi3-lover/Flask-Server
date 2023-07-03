from flask import Flask, request, jsonify
from flask_cors import CORS
from blueprints import user_bp
app = Flask(__name__)
CORS(app, supports_credentials=True)
# 注册蓝图
app.register_blueprint(user_bp)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()

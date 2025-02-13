from flask import Flask, session

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

from controllers import index, data, metrics, predict, diamonds

if __name__ == '__main__':
    app.run(debug=True)
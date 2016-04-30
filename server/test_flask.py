#!/usr/bin/python
from flask import Flask
app = Flask(__name__)

HOST = '166.111.225.1'
PORT = 5000

@app.route('/')
def index():
    return 'Yizhuang ZHOU\'s project. ( under construction )'

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)

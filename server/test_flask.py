#!/usr/bin/python
from flask import Flask
app = Flask(__name__)

HOST = '166.111.225.1'
PORT = 5000

@app.route('/')
def index():
    return 'Yizhuang ZHOU\'s project. ( under construction )'

@app.route('/reid')
def reid():
    return 'Request: person image, Response: similarity to database.'

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)

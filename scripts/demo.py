import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask
import flask
import webbrowser
import os

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

server = Flask(__name__)
app = dash.Dash(name = __name__, server = server)

app.layout = html.Div( 
   html.Iframe(src='/static/lda_B00MXWFUQC.html', style=dict(position="absolute", left="0", top="0", width="100%", height="100%"))
)

@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:8050/', new=0, autoraise=True) 
    app.run_server(debug=True, use_reloader=False)
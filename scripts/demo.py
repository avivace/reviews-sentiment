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

urls = {
    'plot_1': '/static/lda_B00MXWFUQC.html',
    'plot_2': '/static/lda_B00UCZGS6S.html'
}

init_key, init_val = next(iter(urls.items()))

print(init_key)
print(init_val)

dd = dcc.Dropdown(
    id='dropdown',
    options= [{'label': k, 'value': v} for k, v in urls.items()],
    #value=init_key,
    placeholder="Choose the plot"
)

# embedded plot element whose `src` parameter will
# be populated and updated with dropdown values
plot = html.Iframe(
    id='plot',
    style={'border': 'none', 'width': '100%', 'height': 500},
    src=init_val
)

# set div containing dropdown and embedded plot
app.layout = html.Div(children=[dd, plot])

# update `src` parameter on dropdown select action
@app.callback(
    Output(component_id='plot', component_property='src'),
    [Input(component_id='dropdown', component_property='value')]
)
def update_plot_src(input_value):
    return input_value

'''app.layout = html.Div( 
   html.Iframe(src='/static/lda_B00MXWFUQC.html', style=dict(position="absolute", left="0", top="0", width="100%", height="100%"))
)'''

@app.server.route('/static/<resource>')
def serve_static(resource):
    return flask.send_from_directory(STATIC_PATH, resource)


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:8050/', new=0, autoraise=True) 
    app.run_server(debug=True, use_reloader=False)
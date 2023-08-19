import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import pymongo
import os
from flask import Flask, request, jsonify
from enum import Enum
from collections import Counter, OrderedDict
from bson import ObjectId
import requests, json
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__)

API_BASE_URL = 'http://localhost:8080'

MONGO_USER= 'MONGO_USER'
MONGO_PASSWORD= 'MONGO_PASSWORD'
MONGO_HOST= 'MONGO_HOST'
MONGO_DATABASE= 'MONGO_DATABASE'

class PlotType(Enum):
    PIE = 'pie'
    BAR = 'bar'
    HOURLY_DIST = 'hourly_dist'
    REACTIONS_ALONG_TIME = 'reactions_along_time'
    TOTAL_SCORE_BOX = 'total_score_box'
    PRESS_ACTION_CORRELATION = 'press_action_correlation'
    USER_CHANGE = 'user_change'
    REACTION_FREQUENCY_CHANGE = 'reaction_frequency_change'

# Assuming the dataframe df is already defined
data1 = pd.read_csv("../data/logs_05-23_19-11-48.csv")
data2 = pd.read_csv("../data/logs_05-23_19-12-15.csv")
data3 = pd.read_csv("../data/logs_05-23_19-12-35.csv")
data4 = pd.read_csv("../data/logs_05-23_19-13-23.csv")
df = data2


labels = df['reaction'].unique()
counts = df['reaction'].value_counts().values.tolist()
df['time_label'] = pd.to_datetime(df.timestamp, unit='ms')
df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
df.set_index('time_label', inplace=True)

# API endpoint to retrieve plot information
@app.server.route('/plot', methods=['GET'])
def get_plot_data():
    # TODO: You may add an authentication step in order to refuse to show data to unauthenticated users
    event_id = request.args.get('event')
    plot_type = request.args.get('type', 'pie')
    user_id = request.args.get('user', None)
    reactions = json.loads(request.args.get('reactions', None))

    event_reactions = list(db.eventReactions.find({'eventId': event_id}))
    if plot_type == PlotType.PIE.value:
        reaction_counts = Counter(reaction['reaction'] for reaction in event_reactions)
        plot_data = {
            'labels': list(reaction_counts.keys()),
            'values': list(reaction_counts.values()),
        }

    if plot_type == PlotType.BAR.value:
        user_reaction_counts = Counter(reaction['userId'] for reaction in event_reactions)
        user_reaction_counts = OrderedDict(user_reaction_counts.most_common())
        plot_data = {
            'labels': list(user_reaction_counts.keys()),
            'values': list(user_reaction_counts.values()),
        }

    if plot_type == PlotType.HOURLY_DIST.value:
        query = {'eventId': event_id}
        if reactions:
            query['reaction'] = {'$in': reactions}
        event_reactions = list(db.eventReactions.find(query))

        hourly_dist_data = {}
        for reaction in event_reactions:
            timestamp = int(reaction['timestamp']) / 1000.0
            reaction_hour = datetime.fromtimestamp(timestamp).hour
            
            if reaction_hour not in hourly_dist_data:
                hourly_dist_data[reaction_hour] = []
                
            hourly_dist_data[reaction_hour].append(reaction['reaction'])

        # Prepare data for Plotly
        hist_dict = {}
        for hour, reactions in hourly_dist_data.items():
            for reaction in reactions:
                if reaction not in hist_dict:
                    hist_dict[reaction] = []
                hist_dict[reaction].append(hour)


        for reaction, hours in hist_dict.copy().items():
            if len(hours) < 2:
                hist_dict.pop(reaction)
        
        plot_data = {'labels': list(hist_dict.keys()), 
                     'hist_data': list(hist_dict.values())}

    return jsonify(plot_data)


# Define the app layout
def build_app_layout():
    # users = list(users_collection.find({}, {'userId': 1}))
    users = list(db.users.find({'events': {'$exists': True, '$ne': []}}, {'username': 1}))
    app.layout = html.Div(
        children=[
            html.H1('My Dash Application'),
            dcc.Dropdown(
                id='owner-selection-dropdown',
                options=[{'label': user['username'], 'value': str(user['_id'])} for user in users],
                multi=False,
                placeholder='Select a user',
            ),
            dcc.Dropdown(
                id='event-selection-dropdown',
                multi=False,
                placeholder='Select an event',
                style={'display': 'none'}
            ),
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Bar Plot', 'value': 'bar'},
                    {'label': 'Hourly Distribution of Interaction', 'value': 'hourly_dist'},
                    {'label': 'Change of Number of Reactions', 'value': 'reactions_along_time'},
                    {'label': 'Score Box Plot', 'value': 'total_score_box'},
                    {'label': 'Correlation of Press times and Reaction', 'value': 'press_action_correlation'},
                    {'label': "User's belief Change", 'value': 'user_change'},
                    {'label': "Reaction's Frequency Change", 'value': 'reaction_frequency_change'},
                ],
                value='pie',
                style={'display': 'none'}
            ),
            dcc.Loading(id="loading-icon", children=[html.Div(id="dropdown-container")], type="circle"),
            dcc.Dropdown(
                id='reaction-selection-dropdown',
                options=[],
                multi=True,
                style={'display': 'none'} # Hide dropdown
            ),
            dcc.Dropdown(
                id='user-selection-dropdown',
                options=[],
                multi=False,
                placeholder='Select a user'
            ),
            dcc.Graph(
                id='displayed-plot',
                # style={'display': 'none'},
            ),    
        ]
    )

# Callback to update the event dropdown based on the selected user
@app.callback(
    Output('event-selection-dropdown', 'options'),
    Output('event-selection-dropdown', 'style'),
    Input('owner-selection-dropdown', 'value')
)
def update_event_dropdown(selected_user):
    if selected_user:
        # Fetch events for the selected user from the database
        user_events = db.events.find({'owner': selected_user})
        event_options = [{'label': event['title'], 'value': event['uuid']} for event in user_events]
        return event_options, {'display': 'block'}
    return [], {'display': 'none'}

@app.callback(
    Output('plot-type-dropdown', 'style'),
    Output('reaction-selection-dropdown', 'style'),
    Output('reaction-selection-dropdown', 'value'),
    Output('reaction-selection-dropdown', 'options'),
    Output('user-selection-dropdown', 'style'),
    Output('user-selection-dropdown', 'value'),
    [Input('owner-selection-dropdown', 'value'), Input('plot-type-dropdown', 'value'), Input('event-selection-dropdown', 'value')]
)
def update_plot_dropdown(owner_id, plot_type, event_id):
    if owner_id is None:
        return {'display': 'none'}, {'display': 'none'}, [], [], {'display': 'none'}, None
    if plot_type == 'hourly_dist':
        # Display reaction-selection-dropdown and deselect all options
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$group": {"_id": "$reaction"}},
            {"$project": {"_id": 0, "reaction": "$_id"}}
        ]
        reactions = list(db.eventReactions.aggregate(pipeline))
        return {'display': 'block'}, {'display': 'block'}, [], [{'label': r['reaction'], 'value': r['reaction']} for r in reactions], {'display': 'none'}, None
    elif plot_type == 'user_change':
        # Display user-selection-dropdown and select the first user by default

        # Define the event_id for which you want to find unique users

        # Use MongoDB's aggregation framework to find unique users for the event
        pipeline = [
            {"$match": {"eventId": ObjectId(event_id)}},
            {"$group": {"_id": "$userId"}},
            {"$project": {"userId": "$_id", "_id": 0}}
        ]
        participated_users = list(db.eventReactions.aggregate(pipeline))
        return {'display': 'block'}, {'display': 'none'}, None, [], {'display': 'block'}, participated_users
    else:
        # Hide both dropdowns and clear the selected values
        return {'display': 'block'}, {'display': 'none'}, None, [], {'display': 'none'}, None

@app.callback(
    # Output('displayed-plot', 'style'),
    Output('displayed-plot', 'figure'),
    [Input('event-selection-dropdown', 'value'),
     Input('plot-type-dropdown', 'value'),
     Input('reaction-selection-dropdown', 'value'),
     Input('user-selection-dropdown', 'value')]
)
def update_graph(event_id, plot_type, reaction_selection, user_selection):
    if event_id is None:
        return {}

    emoji_mapping = {
        '🙁': 1,
        '👎': 2,
        '🙄': 3,
        '🙂': 4,
        '😆': 5,
        '👍': 6,
        '👏': 7,
        '😍': 8,
    }

    url = (f'{API_BASE_URL}/plot')
    params = {
        'type': plot_type,
        'event': event_id,
        'user': user_selection,
        'reactions': json.dumps(reaction_selection),
    }
    if plot_type in ['pie', 'bar', 'hourly_dist']:
        # print(requests.get(url, params=params).text)
        data = json.loads(requests.get(url, params=params).text)

    if plot_type == 'pie':
        labels, values = data['labels'], data['values']
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title_text='Pie Chart of Reactions')
        return fig
    elif plot_type == 'bar':
        labels, values = data['labels'], data['values']
        fig = go.Figure(data=[go.Bar(y=values, hovertext=labels)])
        fig.update_layout(title_text='Bar Plot of User Reactions', yaxis_title="Reaction Counts")
        return fig
    elif plot_type=="hourly_dist":
        labels, hist_data = data['labels'], data['hist_data']
        fig = ff.create_distplot(hist_data, labels, bin_size=.2, curve_type='kde', show_hist=False,)
        fig.update_layout(title_text='Hourly Distribution for Selected Reactions')
        return fig
    elif plot_type=="reactions_along_time":
        data_count = df.groupby(df['timestamp'].dt.date).count()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_count.index, y=data_count['reaction'], mode='lines'))

        fig.update_layout(title_text='Number of Reactions Over Time (daily)',
                        xaxis_title='Date',
                        yaxis_title='Number of Reactions')
        return fig
    elif plot_type == "total_score_box":
        df['reaction_value'] = df['reaction'].map(emoji_mapping).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Box(x=df['reaction_value'], name='Reactions', orientation='h'))
        fig.update_layout(
            title='Distribution of Reactions',
            xaxis=dict(title='Reaction Value')
        )
        return fig
    elif plot_type == "press_action_correlation":
        df['reaction_numeric'] = df['reaction'].map(emoji_mapping)
        # Calculate the average number of presses per reaction for each user
        average_presses = df.groupby(['userId', 'reaction_numeric']).size().groupby('userId').mean().reset_index(name='average_presses')
        # Merge this information back into the dataframe
        df_ = pd.merge(df, average_presses, how='left', on='userId')
        # Create hexbin plot
        fig = go.Figure(go.Histogram2d(
            x=df_['reaction_numeric'],
            y=df_['average_presses'],
            autobinx=False,
            xbins=dict(start=0, end=8, size=1),  # Adjust size to change the resolution of the hexbins
            autobiny=False,
            ybins=dict(start=0, end=df_['average_presses'].max(), size=df_['average_presses'].max()/20),  # Adjust size to change the resolution of the hexbins
            colorscale='Viridis'
        ))
        fig.update_layout(
            title='Hexbin Plot of Average Button Presses vs Reaction Numeric Value',
            xaxis_title='Reaction Numeric Value',
            yaxis_title='Average Button Presses',
            autosize=False,
            width=600,
            height=600,
        )
        return fig
    elif plot_type=="user_change":
        user_id = user_selection
        user_data = df[df['userId'] == user_id]
        # Apply the mapping function to the reaction column
        user_data['reaction_value'] = user_data['reaction'].map(emoji_mapping).fillna(0)
        # Plot the change in reaction over time using Plotly
        fig = px.line(user_data, x='timestamp', y='reaction_value', title=f'Reaction Change Over Time for User ID: {user_id}')
        fig.update_xaxes(title_text='Timestamp')
        fig.update_yaxes(title_text='Reaction Value')
        return fig
    elif plot_type=="reaction_frequency_change":
        weekly_counts = df.resample('W').reaction.value_counts()
        # unstack the hierarchical index produced by value_counts
        weekly_counts = weekly_counts.unstack(level=-1, fill_value=0)
        fig = go.Figure()
        # Loop through each unique reaction
        for reaction in weekly_counts.columns:
            fig.add_trace(go.Scatter(x=weekly_counts.index, 
                                    y=weekly_counts[reaction], 
                                    mode='lines', 
                                    name=reaction))
        fig.update_layout(title='Reaction frequency over time',
                        xaxis_title='Time',
                        yaxis_title='Frequency')

        return fig

if __name__ == '__main__':
    env_vars = [MONGO_USER, MONGO_PASSWORD, MONGO_HOST, MONGO_DATABASE]
    missing = set(env_vars) - set(os.environ)
    if missing:
        raise Exception("Environment variables do not exist: %s" % missing)

    mongo_uri = "mongodb://%s:%s@%s/%s" % (os.environ[MONGO_USER], 
                                           os.environ[MONGO_PASSWORD],
                                           os.environ[MONGO_HOST],
                                           os.environ[MONGO_DATABASE])
    client = pymongo.MongoClient(mongo_uri)
    db = client.get_database(os.environ[MONGO_DATABASE])

    build_app_layout()

    app.run_server(debug=True, port=8080)

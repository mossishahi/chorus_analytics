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
from flask_cors import CORS, cross_origin

# Initialize the Dash app
app = dash.Dash(__name__)

API_BASE_URL = 'http://localhost:8088'

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

# API endpoint to retrieve plot information
@app.server.route('/plot', methods=['GET'])
@cross_origin()
def get_plot_data():
    # TODO: You may add an authentication step in order to refuse to show data to unauthenticated users
    event_id = request.args.get('event')
    plot_type = request.args.get('type', 'pie')
    user_id = request.args.get('user', None)
    reactions = json.loads(request.args.get('reactions', None)) if request.args.get('reactions', None) else None
    emoji_mapping = json.loads(request.args.get('mapping', None)) if request.args.get('mapping', None) else None

    event_reactions = list(db.eventreactions.find({'eventId': event_id}))
    if plot_type == PlotType.PIE.value:
        reaction_counts = Counter(reaction['message'] for reaction in event_reactions)
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
            query['message'] = {'$in': reactions}
        event_reactions = list(db.eventreactions.find(query))

        hourly_dist_data = {}
        for reaction in event_reactions:
            timestamp = int(reaction['timestamp']) / 1000.0
            reaction_hour = datetime.fromtimestamp(timestamp).hour
            
            if reaction_hour not in hourly_dist_data:
                hourly_dist_data[reaction_hour] = []
                
            hourly_dist_data[reaction_hour].append(reaction['message'])

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
        
    if plot_type == PlotType.REACTIONS_ALONG_TIME.value:
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$addFields": {"date": {"$toDate": {"$add": [0, "$timestamp"]}}}},
            {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date"}}, "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        result = list(db.eventreactions.aggregate(pipeline))

        # Prepare data for Plotly
        plot_data = {
            'labels': [item['_id'] for item in result],
            'values': [item['count'] for item in result]
        }

    if plot_type == PlotType.TOTAL_SCORE_BOX.value:
        pipeline = [
            {"$match": {"eventId": event_id}}
        ]
        event_reactions = list(db.eventreactions.aggregate(pipeline))

        # Prepare data for Plotly
        for reaction in event_reactions:
            reaction['reaction_value'] = emoji_mapping.get(reaction['message'], 0)
        plot_data = {
            'values': [r['reaction_value'] for r in event_reactions]
        }

    if plot_type == PlotType.PRESS_ACTION_CORRELATION.value:
        # Fetch reactions associated with the event from the MongoDB collection
        pipeline = [
            {"$match": {"eventId": event_id}}
        ]
        event_reactions = list(db.eventreactions.aggregate(pipeline))

        # Apply the mapping to each reaction and create a new field 'reaction_numeric'
        for reaction in event_reactions:
            reaction['reaction_numeric'] = emoji_mapping.get(reaction['message'], 0)

        # Calculate the average number of presses per reaction for each user
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$group": {"_id": {"userId": "$userId", "reaction_numeric": "$reaction_numeric"}, "count": {"$sum": 1}}},
            {"$group": {"_id": "$_id.userId", "average_presses": {"$avg": "$count"}}}
        ]
        average_presses_result = list(db.eventreactions.aggregate(pipeline))

        # Merge this information back into the event_reactions list
        for user_data in average_presses_result:
            user_id = user_data['_id']
            average_presses = user_data['average_presses']
            for reaction in event_reactions:
                if reaction['userId'] == user_id:
                    reaction['average_presses'] = average_presses

        # Extract reaction_numeric and average_presses from event_reactions
        reaction_numeric_values = [reaction['reaction_numeric'] for reaction in event_reactions]
        average_presses_values = [reaction.get('average_presses', 0) for reaction in event_reactions]

        plot_data = {
            'reaction_numeric': reaction_numeric_values,
            'average_presses': average_presses_values,
        }

    if plot_type == PlotType.USER_CHANGE.value:
        if user_id is None: return {}
        
        pipeline = [
            {"$match": {"eventId": event_id, "userId": user_id}},
            {"$sort": {"timestamp": 1}},
        ]
        user_event_reactions = list(db.eventreactions.aggregate(pipeline))

        # Apply the mapping to each reaction and create a new field 'reaction_value'
        for reaction in user_event_reactions:
            reaction['reaction_value'] = emoji_mapping.get(reaction['message'], 0)

        # Prepare data for Plotly
        plot_data = {
            'timestamps': [datetime.fromtimestamp(reaction['timestamp']/1000.0) for reaction in user_event_reactions],
            'reaction_values': [reaction['reaction_value'] for reaction in user_event_reactions],
        }

    if plot_type == PlotType.REACTION_FREQUENCY_CHANGE.value:
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$addFields": {"week": {"$week": {"date": {"$toDate": "$timestamp"}}}}},
            {"$group": {
                "_id": {"week": "$week", 'message': "$message"},
                "count": {"$sum": 1}
            }},
            {"$group": {
                "_id": "$_id.week",
                "counts": {"$push": {'message': "$_id.message", "count": "$count"}}
            }}
        ]
        result = list(db.eventreactions.aggregate(pipeline))

        # Prepare data for Plotly
        x_vals = [item['_id'] for item in result]
        reaction_data = {reaction['message']: [0] * len(x_vals) for item in result for reaction in item['counts']}
        for item in result:
            for reaction_count in item['counts']:
                reaction_data[reaction_count['message']][x_vals.index(item['_id'])] = reaction_count['count']

        # Prepare data for Plotly
        plot_data = {
            'x_vals': x_vals,
            'reaction_data': reaction_data,
        }

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
                options=[{'label': user['username'], 'value': user['username']} for user in users],
                multi=False,
                placeholder='Select a user',
            ),
            dcc.Dropdown(
                id='event-selection-dropdown',
                multi=False,
                placeholder='Select an event',
                style={'display': 'none'}
            ),
            dcc.Tabs(
                id='plot-type-tabs',
                value='pie',  # Initial tab value
                children=[
                    dcc.Tab(label='Pie Chart', value='pie'),
                    dcc.Tab(label='Bar Plot', value='bar'),
                    dcc.Tab(label='Hourly Distribution', value='hourly_dist'),
                    dcc.Tab(label='Reactions Over Time', value='reactions_along_time'),
                    dcc.Tab(label='Score Box Plot', value='total_score_box'),
                    dcc.Tab(label='Press-Action Correlation', value='press_action_correlation'),
                    dcc.Tab(label="User's Belief Change", value='user_change'),
                    dcc.Tab(label="Reaction Frequency Change", value='reaction_frequency_change'),
                ]
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
        event_options = [{'label': event['title'], 'value': str(event['_id'])} for event in user_events]
        return event_options, {'display': 'block'}
    return [], {'display': 'none'}

@app.callback(
    Output('plot-type-tabs', 'style'),
    Output('reaction-selection-dropdown', 'style'),
    Output('reaction-selection-dropdown', 'value'),
    Output('reaction-selection-dropdown', 'options'),
    Output('user-selection-dropdown', 'style'),
    Output('user-selection-dropdown', 'options'),
    [Input('owner-selection-dropdown', 'value'), Input('plot-type-tabs', 'value'), Input('event-selection-dropdown', 'value')]
)
def update_plot_dropdown(owner_id, plot_type, event_id):
    if owner_id is None:
        return {'display': 'none'}, {'display': 'none'}, [], [], {'display': 'none'}, []
    if plot_type == 'hourly_dist':
        # Display reaction-selection-dropdown and deselect all options
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$group": {"_id": "$message"}},
            {"$project": {"_id": 0, 'message': "$_id"}}
        ]
        reactions = list(db.eventreactions.aggregate(pipeline))
        return {'display': 'block'}, {'display': 'block'}, [], [{'label': r['message'], 'value': r['message']} for r in reactions], {'display': 'none'}, []
    elif plot_type == 'user_change':
        # Display user-selection-dropdown and select the first user by default

        # Define the event_id for which you want to find unique users

        # Use MongoDB's aggregation framework to find unique users for the event
        pipeline = [
            {"$match": {"eventId": event_id}},
            {"$group": {"_id": "$userId"}},
            {"$project": {"userId": "$_id", "_id": 0}}
        ]
        participated_users = list(db.eventreactions.aggregate(pipeline))
        return {'display': 'block'}, {'display': 'none'}, None, [], {'display': 'block'}, [{'label': k['userId'], 'value': k['userId']} for k in participated_users]
    else:
        # Hide both dropdowns and clear the selected values
        return {'display': 'block'}, {'display': 'none'}, None, [], {'display': 'none'}, []

@app.callback(
    # Output('displayed-plot', 'style'),
    Output('displayed-plot', 'figure'),
    [Input('event-selection-dropdown', 'value'),
     Input('plot-type-tabs', 'value'),
     Input('reaction-selection-dropdown', 'value'),
     Input('user-selection-dropdown', 'value')]
)
def update_graph(event_id, plot_type, reaction_selection, user_selection):
    if event_id is None:
        return {}

    emoji_mapping = { '🙁': 1, '👎': 2, '🙄': 3, '🙂': 4, '😆': 5, '👍': 6, '👏': 7, '😍': 8 }
    url = (f'{API_BASE_URL}/plot')
    params = {
        'type': plot_type,
        'event': event_id,
        'user': user_selection,
        'reactions': json.dumps(reaction_selection),
        'mapping': json.dumps(emoji_mapping),
    }
    data = json.loads(requests.get(url, params=params).text)

    if plot_type == PlotType.PIE.value:
        labels, values = data['labels'], data['values']
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title_text='Pie Chart of Reactions')
        return fig
    elif plot_type == PlotType.BAR.value:
        labels, values = data['labels'], data['values']
        fig = go.Figure(data=[go.Bar(y=values, hovertext=labels)])
        fig.update_layout(title_text='Bar Plot of User Reactions', yaxis_title="Reaction Counts")
        return fig
    elif plot_type == PlotType.HOURLY_DIST.value:
        labels, hist_data = data['labels'], data['hist_data']
        fig = ff.create_distplot(hist_data, labels, bin_size=.2, curve_type='kde', show_hist=False,)
        fig.update_layout(title_text='Hourly Distribution for Selected Reactions')
        return fig
    elif plot_type == PlotType.REACTIONS_ALONG_TIME.value:
        labels, values = data['labels'], data['values']
        if len(labels) < 2:
            return {}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels, y=values, mode='lines'))

        fig.update_layout(title_text='Number of Reactions Over Time (daily)',
                        xaxis_title='Date',
                        yaxis_title='Number of Reactions')
        return fig
    elif plot_type == PlotType.TOTAL_SCORE_BOX.value:
        values = data['values']
        fig = go.Figure()
        fig.add_trace(go.Box(x=values, name='Reactions', orientation='h'))
        fig.update_layout(
            title='Distribution of Reactions',
            xaxis=dict(title='Reaction Value')
        )
        return fig
    elif plot_type == PlotType.PRESS_ACTION_CORRELATION.value:
        reaction_numeric, average_presses = data['reaction_numeric'], data['average_presses']
        # Create hexbin plot
        fig = go.Figure(go.Histogram2d(
            x=reaction_numeric,
            y=average_presses,
            autobinx=False,
            xbins=dict(start=0, end=8, size=1),  # Adjust size to change the resolution of the hexbins
            autobiny=False,
            ybins=dict(start=0, end=max(average_presses), size=max(average_presses)/20),  # Adjust size to change the resolution of the hexbins
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
    elif plot_type == PlotType.USER_CHANGE.value:
        if user_selection is None:
            return {}
        timestamps, reaction_values = data['timestamps'], data['reaction_values']
        # Plot the change in reaction over time using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=reaction_values, mode='lines', name='Reaction Value'))
        # fig = px.line([timestamps, reaction_values], x='timestamp', y='reaction_value', title=f'Reaction Change Over Time for User ID: {user_selection}')
        fig.update_xaxes(title_text='Timestamp')
        fig.update_yaxes(title_text='Reaction Value')
        return fig
    elif plot_type == PlotType.REACTION_FREQUENCY_CHANGE.value:
        x_vals, reaction_data = data['x_vals'], data['reaction_data']
        fig = go.Figure()
        # Loop through each unique reaction
        for reaction in reaction_data:
            fig.add_trace(go.Scatter(x=x_vals, 
                                     y=reaction_data[reaction], 
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

    app.run_server(debug=True, port=8088)

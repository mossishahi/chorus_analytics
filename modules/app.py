import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

data1 = pd.read_csv("../data/logs_05-23_19-11-48.csv")
data2 = pd.read_csv("../data/logs_05-23_19-12-15.csv")
data3 = pd.read_csv("../data/logs_05-23_19-12-35.csv")
data4 = pd.read_csv("../data/logs_05-23_19-13-23.csv")
df = data2

# Initialize the Dash app
app = dash.Dash(__name__)

# Assuming the dataframe df is already defined
labels = df['reaction'].unique()
counts = df['reaction'].value_counts().values.tolist()
df['time_label'] = pd.to_datetime(df.timestamp, unit='ms')
df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
df.set_index('time_label', inplace=True)

# Define the app layout

app.layout = html.Div(
    children=[
        html.H1('My Dash Application'),
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
            value='pie'
        ),
        dcc.Loading(id="loading-icon", children=[html.Div(id="dropdown-container")], type="circle"),
        dcc.Dropdown(
            id='reaction-selection-dropdown',
            options=[{'label': reaction, 'value': reaction} for reaction in df['reaction'].unique()],
            multi=True,
            style={'display': 'none'} # Hide dropdown
        ),
        dcc.Dropdown(
            id='user-selection-dropdown',
            options=[{'label': user_id, 'value': user_id} for user_id in df['userId'].unique() if len(df[df['userId']==user_id])>1],
            multi=False,
            placeholder='Select a user'
        ),
        dcc.Graph(id='displayed-plot'),    
    ]
)



@app.callback(
    Output('reaction-selection-dropdown', 'style'),
    Output('reaction-selection-dropdown', 'value'),
    Output('user-selection-dropdown', 'style'),
    Output('user-selection-dropdown', 'value'),
    [Input('plot-type-dropdown', 'value')]
)
def update_dropdown(plot_type):
    if plot_type == 'hourly_dist':
        # Display reaction-selection-dropdown and deselect all options
        return {'display': 'block'}, [], {'display': 'none'}, None
    elif plot_type == 'user_change':
        # Display user-selection-dropdown and select the first user by default
        return {'display': 'none'}, None, {'display': 'block'}, df['userId'].unique()[0]
    else:
        # Hide both dropdowns and clear the selected values
        return {'display': 'none'}, None, {'display': 'none'}, None


@app.callback(
    Output('displayed-plot', 'figure'),
    [Input('plot-type-dropdown', 'value'),
     Input('reaction-selection-dropdown', 'value'),
     Input('user-selection-dropdown', 'value')]
)

def update_graph(plot_type, reaction_selection, user_selection):
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
    if plot_type == 'pie':
        fig = go.Figure(data=[go.Pie(labels=labels, values=counts)])
        fig.update_layout(title_text='Pie Chart of Reactions')
        return fig
    elif plot_type == 'bar':
        user_reaction_counts = df['userId'].value_counts()
        fig = go.Figure(data=[go.Bar(y=user_reaction_counts.values,
                                     hovertext=user_reaction_counts.index.astype(str))])
        fig.update_layout(title_text='Bar Plot of User Reactions', yaxis_title="Reaction Counts")
        return fig
    elif plot_type=="hourly_dist":
        selected_reactions = reaction_selection or df['reaction'].unique()
        df_selected = df[df['reaction'].isin(selected_reactions)]
        df_selected.index = pd.to_datetime(df_selected.index)
        # Create an 'hour' column
        df_selected['hour'] = df_selected.index.hour
        hist_data = []
        group_labels = []
        # Get all unique reactions
        unique_reactions = df_selected['reaction'].unique()
        # Loop through all unique reactions
        for reaction in unique_reactions:
            # get the hours where the reaction occurred
            reaction_hours = df_selected[df_selected['reaction'] == reaction].hour
            # Only add to plot if there are at least two unique values
            if len(reaction_hours.unique()) > 1:
                hist_data.append(reaction_hours)
                group_labels.append(reaction)

        # Create distplot with curve_type set to 'normal'
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, curve_type='kde', show_hist=False,)
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

    # Remaining code for other plots as before

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
    # html_string = app.index_string

    # # Save the HTML string to a file
    # with open('dashboard.html', 'w') as file:
    #     file.write(html_string)

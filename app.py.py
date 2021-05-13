#!/usr/bin/env python
# coding: utf-8

# In[47]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.pyplot as plt
import base64

# In[48]:

server = app.server
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

##################
# Controls
##################

# This creates the gray column with the slider on the left

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Data Togglers", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select Date Range", className="lead"),
        html.P(
            "(Use the two drag bars to select the date range)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        html.Div(id='time_slider', style={"marginBottom": 80, "font-size": 12}),
        # USE THIS TO CHANGE HOW LOW YOU WANT THE SIDE BAR TO GO
        html.Label("Narrow by Indication and Phase of Development", className="lead"),
        html.P(
            "(Use the two drag bars to select the date range)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),

        dbc.Label("Select Indications"),
        dcc.Dropdown(id='indication_selections',
                     placeholder='None',
                     multi=True),

        dbc.Label("Select Phase"),
        dcc.Dropdown(id='phase_selections',
                     placeholder='None',
                     multi=True),

        dbc.Label("Select Sponsor Type"),
        dcc.Dropdown(id='sponsor_type',
                     placeholder='None',
                     multi=True)

    ]
)

# This creates the graph
ClINICAL_TRIALS_DATA = [
    dbc.CardHeader(html.H5("Clinical Trials Over This Time Period")),
    dbc.CardBody(
        [dbc.Alert(
            "Not enough data to render this plot, please adjust the filters",
            id="no-data",
            color="warning",
            style={"display": "none"},
        ),
            html.Div(dcc.Graph(id='timeline', className="dash-bootstrap"),
                     style={"marginBottom": 80, "font-size": 12}),
        ],

    )
]

ClINICAL_TRIALS_METRICS = [
    dbc.CardHeader(html.H5("Study Characteristics")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-clin-data",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="nont-enough-data",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.CardGroup([
                        dbc.Card(dcc.Graph(id='phase-bars')),
                        dbc.Card(dcc.Graph(id='company-pie')),
                        dbc.Card(dcc.Graph(id='sponsor-graph')),
                        dbc.Card(dcc.Graph(id='indication-bar')),
                    ]),
                    dbc.CardHeader(html.H5("Trial Averages")),
                    dbc.CardGroup([
                        dbc.Card(dcc.Graph(id='study-size')),
                        dbc.Card(dcc.Graph(id='study-duration'))])
                ],
                style={"marginTop": 1, "marginBottom": 1, "width": 1},
            ),
        ]
    )]

BODY = dbc.Container(
    [dbc.Row([
        dbc.Col(LEFT_COLUMN, md=2, align="top"),  # Use the brackets to keep them in the same row
        dbc.Col(dbc.Card(ClINICAL_TRIALS_DATA), md=10),
    ],
        style={"marginTop": 20},
    ),
        dbc.Row([
            dbc.Col(ClINICAL_TRIALS_METRICS, md=12)],
            className="mt-12", )

    ], fluid=True)

# In[49]:


####################
# App
####################

app.layout = html.Div([html.Div([
    html.Label("Enter the year the product enters the market"),
    dcc.Input(id='dz', type="text", debounce=True, placeholder='lupus'),
    html.Label("Enter the earliest date"),
    dcc.Input(id='earl', type="text",
              debounce=True, placeholder='earliest date, yyyy-mm-dd: '),
    html.Label("Enter the latest date"),
    dcc.Input(id='late', type="text",
              debounce=True, placeholder='latest date, yyyy-mm-dd: '),
    html.Button('SEARCH', id='search'),
    dcc.Store(id='trials'),
    html.Div(id='studies'),
    html.Div(children=[BODY])
])
])


# In[50]:


############
# Call backs to activate the clinicaltrials.gov api and store data
###########
@app.callback(
    Output('trials', 'data'),
    [Input('search', 'n_clicks'),
     Input('dz', 'value'),
     Input('earl', 'value'),
     Input('late', 'value')]
)
def get_data(n_clicks, dz, earl, late):
    if n_clicks is None:
        raise PreventUpdate
    else:
        disease = dz

        url = 'https://clinicaltrials.gov/api/query/study_fields?expr={}&fmt=JSON&type=Intr&max_rnk=999&fields=NCTId,Condition,BriefTitle,OrgFullName,LeadSponsorClass,StartDate, PrimaryCompletionDate,PrimaryOutcomeMeasure,InterventionDescription,Phase,InterventionName,InterventionType,DetailedDescription,EnrollmentCount,CentralContactName,CentralContactEMail'.format(
            disease)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml')
        elements = soup.find("p").text

        data = json.loads(elements)
        data_list = data['StudyFieldsResponse']['StudyFields']
        clinicalgov_df = pd.DataFrame(data_list)
        clinicalgov_df = clinicalgov_df.drop(columns='Rank')
        clinicalgov_df = clinicalgov_df.apply(lambda x: x.str[0])
        clinicalgov_df.sort_values(by=['Phase'], inplace=True, ascending=False)
        clinicalgov_df['Phase'] = clinicalgov_df['Phase'].astype(
            str)  # for some reason they're floats, turning to strings
        clinicalgov_df = clinicalgov_df[
            ~clinicalgov_df.Phase.str.contains('Phase 4')]  # this is likely repurposing, or other stuff not interesting
        # clinicalgov_df = clinicalgov_df[~clinicalgov_df.Phase.str.contains('Not Applicable')] #obviously
        # clinicalgov_df = clinicalgov_df[~clinicalgov_df.Phase.str.contains('nan')]# obviously
        # clinicalgov_df = clinicalgov_df[~clinicalgov_df.Phase.str.contains('Early Phase 1')]  #too eary to be relevant
        # clinicalgov_df = clinicalgov_df[clinicalgov_df.InterventionType.isin(['Drug', 'Biological'])] #Only keeps drugs, and biologics in the dataframe, drop all other intervention types

        clinicalgov_df['ph_num'] = clinicalgov_df.Phase.str.extract('(\d+)')  # extract numeric of phases
        clinicalgov_df['ph_num'] = clinicalgov_df['ph_num'].astype(float)
        clinicalgov_df['name_phase'] = [' '.join(i) for i in
                                        zip(clinicalgov_df['InterventionName'].map(str), clinicalgov_df['Phase'])]
        # clinicalgov_df['name_phase'] = [' '.join(i) for i in zip(clinicsalgov_df['name_phase'].map(str), clinicalgov_df['OrgFullName'])]
        earliest = earl
        latest = late
        clinicalgov_df['StartDate'] = pd.to_datetime(clinicalgov_df['StartDate'])
        clinicalgov_df['PrimaryCompletionDate'] = pd.to_datetime(
            clinicalgov_df['PrimaryCompletionDate'])  # --converts dates to time stamp
        clinicalgov_dff = clinicalgov_df[
            (clinicalgov_df['PrimaryCompletionDate'] > earliest) & (clinicalgov_df['PrimaryCompletionDate'] < latest)]
        clinicalgov_dff['level'] = np.tile([-60, 60, -50, 50, -40, 40, -30, 30, -10, 10, -5, 5, -1, 1],
                                           int(np.ceil(len(clinicalgov_dff['PrimaryCompletionDate']) / 14)))[
                                   :len(clinicalgov_dff['PrimaryCompletionDate'])]
        # Reset the index of the clinicalgov_dff by date
        clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

        return clinicalgov_dff.to_dict()


# In[51]:


####################################
# Callbacks - Display a study to show the data request worked
####################################
@app.callback(
    Output("studies", "children"),
    [Input("trials", "data")])
def update_output(data):
    dff = pd.DataFrame(data)
    return dff.iloc[2, 2]


# In[52]:


@app.callback(Output('indication_selections', 'options'),
              Output('sponsor_type', 'options'),
              Output('phase_selections', 'options'),
              Input('trials', 'data'))
def create_left_col(data):
    ##########################################################
    # THESE ARE THE FILTER VALUES USED LATER TO FILTER THE DATA
    #########################################################
    clinicalgov_dff = pd.DataFrame(data)
    indications = clinicalgov_dff['Condition'].dropna().sort_values(ascending=False).unique()
    indication_selections = [{'label': i, 'value': i} for i in indications]
    sponsors = clinicalgov_dff['LeadSponsorClass'].unique()
    sponsor_type = [{'label': i, 'value': i} for i in sponsors]
    phases = clinicalgov_dff['Phase'].sort_values(ascending=False).unique()
    phase_selections = [{'label': i, 'value': i} for i in phases]
    return indication_selections, sponsor_type, phase_selections


# In[53]:


@app.callback(Output('time_slider', 'children'),
              Input('trials', 'data'))
def timeline_data(data):
    clinicalgov_dff = pd.DataFrame(data)
    numdate = [x for x in range(len(clinicalgov_dff['PrimaryCompletionDate'].sort_values(ascending=True)))]

    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True)
    slider = dcc.RangeSlider(id='time-slider',
                             updatemode='drag',
                             allowCross=False,
                             min=numdate[0],  # the first date
                             max=numdate[-1],  # the last date
                             value=[5 - 5, 5],
                             )
    return slider


# In[54]:


############################################################
# Callbacks - Clinical Trials Over This Time Period
############################################################
@app.callback(
    Output('timeline', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_timeline(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

    # None of the 3 are selected
    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]

    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff

    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    fig = px.scatter(dff,
                     x='PrimaryCompletionDate',
                     y='level',
                     text='name_phase',
                     hover_data=['InterventionName', 'Phase', 'Condition', 'OrgFullName', 'NCTId'],
                     color=dff['Phase'],
                     color_discrete_map={'Phase 1': 'lightcyan', '{Phase 2}': 'royalblue', 'Phase 3': 'darkblue'}
                     )

    fig.update_traces(textposition='top center', textfont=dict(family="sans serif"), textfont_size=14)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(visible=False, showticklabels=False)

    return fig


# In[55]:


@app.callback(
    Output('phase-bars', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value')])
def create_phase_bars(data, value):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]
    phase_dict = dict({'Phase 1': int(len(dff[dff['Phase'].str.contains('1')])),
                       'Phase 2': int(len(dff[dff['Phase'].str.contains('2')])),
                       'Phase 3': int(len(dff[dff['Phase'].str.contains('3')]))})

    phase_clinicalgov_df = pd.DataFrame.from_dict(phase_dict, orient='index', columns=['Count'])
    fig = px.bar(phase_clinicalgov_df,
                 x=phase_clinicalgov_df.index,
                 y='Count',
                 color=phase_clinicalgov_df.index, color_discrete_map={'Phase 1': 'lightcyan',
                                                                       '{Phase 2}': 'royalblue', 'Phase 3': 'darkblue'})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# In[56]:


@app.callback(
    Output('company-pie', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_company_pie(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]
    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff
    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    company_counts = dff.groupby('OrgFullName')['NCTId'].nunique().reset_index()
    company_counts = company_counts.rename(columns={'OrgFullName': 'Company', 'NCTId': 'Study Counts'})
    company_counts = company_counts.sort_values(by='Study Counts', ascending=False)

    fig = px.pie(company_counts,
                 values='Study Counts',
                 names='Company')
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='outside', textinfo='text + label ', hole=.4)

    return fig


# In[57]:


@app.callback(
    Output('sponsor-graph', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_sponsor_pie(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]
    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff
    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    sponsor_counts = dff.groupby('LeadSponsorClass')['NCTId'].nunique().reset_index()
    sponsor_counts = sponsor_counts.rename(columns={'LeadSponsorClass': 'Sponsor', 'NCTId': 'Study Counts'})
    sponsor_counts = sponsor_counts.sort_values(by='Study Counts', ascending=False)

    fig = px.pie(sponsor_counts,
                 values='Study Counts',
                 names='Sponsor')
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition='outside', textinfo='text + label ', hole=.4)

    return fig


# In[58]:


@app.callback(
    Output('indication-bar', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_indication_bar(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)

    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]
    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff
    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    indication_counts = dff.groupby('Condition')['NCTId'].nunique().reset_index()
    indication_counts = indication_counts.rename(columns={'Condition': 'Disease', 'NCTId': 'Study Counts'})
    indication_counts = indication_counts.sort_values(by='Study Counts', ascending=True)

    fig = px.bar(indication_counts,
                 x='Study Counts',
                 y='Disease',
                 color=indication_counts.index,
                 orientation='h')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(visible=True, showticklabels=True)
    fig.update_layout(coloraxis_showscale=False)

    return fig


# In[59]:


@app.callback(
    Output('study-size', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_study_size_bar(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)
    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]

    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff
    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    study_size = dff
    study_size['EnrollmentCount'] = study_size['EnrollmentCount'].astype(float)
    study_size = round(study_size.groupby('Phase')['EnrollmentCount'].mean(), 1).reset_index()
    study_size = study_size.rename(columns={'Phase': 'Phase', 'EnrollmentCount': '# Participants'})
    study_size = study_size.sort_values(by='Phase', ascending=False)
    fig = px.bar(study_size,
                 x='# Participants',
                 y='Phase',
                 color=study_size['Phase'],
                 color_discrete_map={'Phase 1': 'lightcyan',
                                     '{Phase 2}': 'royalblue', 'Phase 3': 'darkblue'},
                 orientation='h')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(visible=True, showticklabels=True)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(font_color='yellow')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'})
    fig.update_xaxes(showgrid=False)

    return fig


# In[60]:


@app.callback(
    Output('study-duration', 'figure'),
    [Input('trials', 'data'),
     Input('time-slider', 'value'),
     Input('indication_selections', 'value'),
     Input('phase_selections', 'value'),
     Input('sponsor_type', 'value')])
def create_study_duration_bar(data, value, indication_selections, phase_selections, sponsor_type):
    clinicalgov_dff = pd.DataFrame(data)

    # Converts dates to time stamp
    clinicalgov_dff['PrimaryCompletionDate'] = pd.to_datetime(clinicalgov_dff['PrimaryCompletionDate'])
    clinicalgov_dff['StartDate'] = pd.to_datetime(clinicalgov_dff['StartDate'])
    clinicalgov_dff = clinicalgov_dff.sort_values(by='PrimaryCompletionDate', ascending=True).reset_index(drop=True)
    dff = clinicalgov_dff[(clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                clinicalgov_dff['PrimaryCompletionDate'].index <= value[1])]

    # Right now there are 3 options, so 2 (is none or is not none) x 2 x 2 means there should be 8 options
    # None of the 3 are selected
    if indication_selections is None and phase_selections is None and sponsor_type is None:
        dff = dff
    # All 3 are selected
    if indication_selections is not None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                      clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor Only
    if indication_selections is None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Sponsor and Indication
    if indication_selections is not None and phase_selections is None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
        # Sponsor and Phase
    if indication_selections is None and phase_selections is not None and sponsor_type is not None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
            clinicalgov_dff['LeadSponsorClass'].str.contains('|'.join(sponsor_type))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication and Phase
    if indication_selections is not None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
            clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                                              (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                                                  clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Indication Only
    if indication_selections is not None and phase_selections is None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Condition'].str.contains('|'.join(indication_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]
    # Phase only
    if indication_selections is None and phase_selections is not None and sponsor_type is None:
        dff = clinicalgov_dff.loc[(clinicalgov_dff['Phase'].str.contains('|'.join(phase_selections))) & (
                    (clinicalgov_dff['PrimaryCompletionDate'].index >= value[0]) & (
                        clinicalgov_dff['PrimaryCompletionDate'].index <= value[1]))]

    study_duration = dff
    study_duration['Length of Study'] = (study_duration['PrimaryCompletionDate'] - study_duration['StartDate']).astype(
        'timedelta64[ns]')
    study_duration['Length of Study'] = study_duration['Length of Study'].dt.days
    study_duration = study_duration.groupby('Phase')['Length of Study'].mean().reset_index()
    study_duration['Length of Study'] = study_duration['Length of Study'] / (365)
    study_duration = study_duration.sort_values(by='Phase', ascending=False)
    fig = px.bar(study_duration,
                 x='Length of Study',
                 y='Phase',
                 color=study_duration['Phase'],
                 color_discrete_map={'Phase 1': 'lightcyan',
                                     '{Phase 2}': 'royalblue', 'Phase 3': 'darkblue'},
                 orientation='h')
    fig.update_layout(showlegend=False)
    fig.update_yaxes(visible=True, showticklabels=True)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(font_color='yellow')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)'})
    fig.update_xaxes(showgrid=False)

    return fig


# In[61]:


app.run_server(debug=True, use_reloader=False)


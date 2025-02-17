import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import io
from dash import dash_table
from sklearn.metrics import mean_absolute_error
from dash.dependencies import State
import plotly.io as pio
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


# Inisialisasi aplikasi Dash
app = dash.Dash(__name__)

# Menu di sisi kiri
menu = html.Div([
    html.H3("Menu", style={'color': '#1C4E80', 'font-size': '24px', 'text-align': 'center'}),
    html.Button('Grafik Data Asli', id='original-button', style={'width': '100%', 'backgroundColor': '#A5D8DD', 'border': 'none', 'color': '#202020', 'font-size': '18px', 'padding': '10px', 'margin-bottom': '10px'}),
    html.Button('Grafik Data Latih', id='train-button', style={'width': '100%', 'backgroundColor': '#A5D8DD', 'border': 'none', 'color': '#202020', 'font-size': '18px', 'padding': '10px', 'margin-bottom': '10px'}),
    html.Button('Grafik Hasil Prediksi', id='prediction-button', style={'width': '100%', 'backgroundColor': '#A5D8DD', 'border': 'none', 'color': '#202020', 'font-size': '18px', 'padding': '10px', 'margin-bottom': '10px'}),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV', style={'width': '100%', 'backgroundColor': '#A5D8DD', 'border': 'none', 'color': '#202020', 'font-size': '18px', 'padding': '10px', 'margin-bottom': '10px'}),
        multiple=False
    ),
    dcc.Input(id='input-week', type='number', placeholder='Minggu yang akan diprediksi', style={'width': '90%', 'margin-top': '5px', 'margin-bottom': '10px', 'font-size': '16px', 'padding': '10px'}),
    html.Button('Prediksi', id='predict-button', style={'width': '100%', 'backgroundColor': '#EA6A47', 'border': 'none', 'color': '#F1F1F1', 'font-size': '18px', 'padding': '10px'}),
    html.Div(id='output-prediction', style={'margin-top': '20px', 'font-size': 'large'}),
    html.Div(id='output-mae', style={'margin-top': '10px', 'font-weight': 'bold'}),
    html.Div(id='output-accuracy', style={'margin-top': '10px', 'font-weight': 'bold'}),
    html.Div(id='output-rsquare', style={'margin-top': '10px', 'font-weight': 'bold'}),
    html.Div([
        html.Label("Persentase Data Latih:", style={'font-size': '16px', 'margin-bottom': '10px'}),
        dcc.Slider(
            id='data-percentage-slider',
            min=10,
            max=100,
            step=10,
            value=80,
            marks={i: f'{i}%' for i in range(0, 101, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
            
            included=True,  
            vertical=False,
            updatemode="drag",
            className='custom-slider',
        )]
    ),
    html.Button('Download Grafik', id='download-button', style={'width': '100%', 'backgroundColor': '#A5D8DD', 'border': 'none', 'color': '#202020', 'font-size': '18px', 'padding': '10px', 'margin-bottom': '10px'}),
        dcc.Download(id="download-graph"),
    
], className='menu', style={
    'backgroundColor': '#F1F1F1', 
    'padding': '10px', 
    'border-radius': '15px',
    'box-shadow': '2px 2px 5px #202020', 
    'height': '100vh', 
    'font-size': '18px',
    'width': '250px', 
})

# Bagian atas dashboard
header = html.Div([
    html.Img(src='assets/logo.png', style={'height': '50px', 'width': '50px'}),
], style={'display': 'flex', 'align-items': 'center', 'backgroundColor': '#7E909A', 'padding': '5px', 'color': '#F1F1F1', 'border-radius': '5px', 'margin-bottom': '10px'})

# Area utama untuk grafik dan tabel
main_area = html.Div([
    dcc.Graph(id='payload-graph'),
    dash_table.DataTable(id='data-table', columns=[
        {'name': 'Week', 'id': 'week'},
        {'name': 'Total Payload (GB)', 'id': 'total_payload_GB'}
    ], style_table={'margin-top': '20px', 'overflowX': 'auto', 'overflowY': 'auto', 'maxHeight': '400px'},
       style_cell={'textAlign': 'left'},  
       page_size=10  
    )
], style={'flex-grow': '1', 'padding': '10px', 'backgroundColor': '#A5D8DD', 'border-radius': '5px'})

# Layout keseluruhan
app.layout = html.Div([
    header,
    html.Div([
        menu,
        main_area
    ], style={'display': 'flex', 'flex-grow': '1', 'backgroundColor': '#F1F1F1'})
], style={'padding': '10px'})

@app.callback(
    [Output('output-prediction', 'children'),
     Output('payload-graph', 'figure'),
     Output('data-table', 'data'),
     Output('output-mae', 'children'),
     Output('output-accuracy', 'children'),
     Output('output-rsquare', 'children')],
    [Input('upload-data', 'contents'),
     Input('input-week', 'value'),
     Input('predict-button', 'n_clicks'),
     Input('original-button', 'n_clicks'),
     Input('train-button', 'n_clicks'),
     Input('prediction-button', 'n_clicks'),
     Input('data-percentage-slider', 'value')],
)

def update_output(contents, week, predict_click, original_click, prediction_click, content_type, data_percentage):
    ctx = dash.callback_context

    if contents is None:
        return "Silakan upload file CSV.", go.Figure(), [], "", "", ""

    # Mengambil file dari upload
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    if 'week' not in data.columns or 'total_payload_GB' not in data.columns:
        return "Data tidak valid. Pastikan kolom 'week' dan 'total_payload_GB' ada.", go.Figure(), [], "", "", ""

    # Membuat model regresi linear
    train_size = int(len(data) * data_percentage / 100)
    train_data = data.iloc[:train_size]  
    test_data = data.iloc[train_size:]  


    X_train = train_data[['week']].values
    y_train = train_data['total_payload_GB'].values

    X_test = test_data[['week']].values
    y_test = test_data['total_payload_GB'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi untuk minggu yang diminta
    prediction = None
    if week is not None and ctx.triggered:
        next_week = np.array([[week]])
        prediction = model.predict(next_week)[0]

    # Menghitung MAE (Mean Absolute Error)
    mae_value = mean_absolute_error(y_test, model.predict(X_test))
    
    # Akurasi sebagai persentase
    accuracy_value = 100 - (mae_value / np.mean(y_test)) * 100  
    #R-square
    r_square = model.score(X_test, y_test)
    # Membuat grafik
    fig = go.Figure()
    table_data = data.to_dict('records')  

    if ctx.triggered and 'original-button' in ctx.triggered[0]['prop_id']:
        fig.add_trace(go.Scatter(x=data['week'], y=data['total_payload_GB'], mode='lines+markers',line=dict(color='blue'), marker=dict(color='blue', size=6) )) 
        fig.update_layout(title='Grafik Data Asli',
                          xaxis_title='Minggu',
                          yaxis_title='Total Payload (GB)')

    elif ctx.triggered and 'train-button' in ctx.triggered[0]['prop_id']:
        fig.add_trace(go.Scatter(x=train_data['week'], y=model.predict(X_train), mode='lines', name='Data Latih'))
        fig.update_layout(title='Grafik Data Latih',
                          xaxis_title='Minggu',
                          yaxis_title='Total Payload (GB)')
    elif ctx.triggered and 'prediction-button' in ctx.triggered[0]['prop_id']:
        fig.add_trace(go.Scatter(x=data['week'], y=data['total_payload_GB'], mode='lines+markers',line=dict(color='blue'), marker=dict(color='blue', size=6), name='Data Asli'))
        fig.add_trace(go.Scatter(x=train_data['week'], y=model.predict(X_train), 
                     mode='lines', name='Data Latih', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test_data['week'], y=model.predict(X_test), 
                     mode='lines', name='Data Uji', line=dict(color='grey')))
        fig.add_trace(go.Scatter(x=[week], y=[prediction], mode='markers', name=f'Prediksi Minggu {week}', marker=dict(color='red', size=8)))

        # Ambil week terakhir dari data asli
        last_week = data.iloc[-1]['week']
        last_actual = data.iloc[-1]['total_payload_GB']
        difference = prediction - last_actual
        direction = "Kenaikan" if difference > 0 else "Penurunan"

        # Tambahkan anotasi untuk perbandingan
        
        fig.add_annotation(
            x=0,  
            y=1,  
            xref="paper", 
            yref="paper",
            text=f"Dari Week {last_week} ke Week {week}: {direction} {abs(difference):.2f} GB",
            showarrow=False, 
            font=dict(color="blue", size=10),
            align="left",  
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            borderpad=3,  
            bgcolor="white",  
            opacity=0.9  
        )


        # Tambahkan garis untuk menghubungkan week terakhir dengan prediksi
        fig.add_shape(
            type="line",
            x0=last_week, y0=last_actual,
            x1=week, y1=prediction,
            line=dict(color="blue", width=2, dash="dot"),
        )

        fig.update_layout(title='Grafik Hasil Prediksi',
                          xaxis_title='Minggu',
                          yaxis_title='Total Payload (GB)')
        
    elif ctx.triggered and 'predict-button' in ctx.triggered[0]['prop_id']:
    
        fig.add_trace(go.Scatter(x=data['week'], y=data['total_payload_GB'], mode='lines+markers',line=dict(color='blue'), marker=dict(color='blue', size=6), name='Data Asli'))
        fig.add_trace(go.Scatter(x=train_data['week'], y=model.predict(X_train), 
                     mode='lines', name='Data Latih', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=test_data['week'], y=model.predict(X_test), 
                     mode='lines', name='Data Uji', line=dict(color='grey')))
        fig.add_trace(go.Scatter(x=[week], y=[prediction], mode='markers', name=f'Prediksi Minggu {week}', marker=dict(color='red', size=8)))

        # Ambil week terakhir dari data asli
        last_week = data.iloc[-1]['week']
        last_actual = data.iloc[-1]['total_payload_GB']
        difference = prediction - last_actual
        direction = "Kenaikan" if difference > 0 else "Penurunan"

        # Tambahkan anotasi untuk perbandingan
        
        fig.add_annotation(
            x=0,  
            y=1,  
            xref="paper", 
            yref="paper",
            text=f"Dari Week {last_week} ke Week {week}: {direction} {abs(difference):.2f} GB",
            showarrow=False, 
            font=dict(color="blue", size=10),
            align="left",  
            xanchor="left",
            yanchor="top",
            borderwidth=1,
            borderpad=5,  
            bgcolor="white",  
            opacity=0.9  
        )


        # Tambahkan garis untuk menghubungkan week terakhir dengan prediksi
        fig.add_shape(
            type="line",
            x0=last_week, y0=last_actual,
            x1=week, y1=prediction,
            line=dict(color="blue", width=2, dash="dot"),
        )

        fig.update_layout(title='Grafik Hasil Prediksi',
                          xaxis_title='Minggu',
                          yaxis_title='Total Payload (GB)')
    
    return (f"Prediksi Total Payload (GB) untuk Minggu {week}: {prediction:.2f}" if prediction is not None else "Silakan masukkan minggu yang akan diprediksi."), fig, table_data, f"MAE: {mae_value:.2f}", f"Akurasi: {accuracy_value:.2f}%", f"R-square: {r_square:.2f}"
    
@app.callback(
    Output('download-graph', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('payload-graph', 'figure')]
)
def download_graph(n_clicks, figure):
    if n_clicks is None:
        return dash.no_update 
    
    # Konversi grafik menjadi file gambar (format PNG)
    image_bytes = pio.to_image(figure, format='png', width=800, height=600)
    
    # Mengembalikan file yang dapat diunduh
    return dcc.send_bytes(image_bytes, "grafik_hasil_prediksi.png")

if __name__ == '__main__':
    app.run(debug=True)
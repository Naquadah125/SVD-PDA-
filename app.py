import os
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objs as go
from utils.svd import process_svd
from utils.pca import process_pca

app = Dash(__name__, title="PCA & SVD Face Analysis")

app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'fontFamily': 'Arial'}, children=[
    html.H1("Phân tích & Tái tạo Khuôn mặt (Window 10)", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Khu vực Upload chung
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Kéo thả ảnh hoặc ', html.A('Chọn ảnh')]),
        style={'border': '2px dashed #3498db', 'height': '60px', 'lineHeight': '60px', 'textAlign': 'center', 'borderRadius': '10px'},
        multiple=False
    ),

    # Slider chọn K chung
    html.Div([
        html.Label("Số lượng thành phần (k):", style={'fontWeight': 'bold'}),
        dcc.Slider(id='k-slider', min=1, max=100, step=1, value=20, marks={1:'1', 50:'50', 100:'100'}, tooltip={"placement": "bottom", "always_visible": True})
    ], style={'padding': '20px'}),

    # Tabs chuyển đổi SVD và PCA
    dcc.Tabs(id="algo-tabs", value='tab-svd', children=[
        dcc.Tab(label='SVD Reconstruction (NumPy)', value='tab-svd'),
        dcc.Tab(label='PCA Reconstruction (Scikit-Learn)', value='tab-pca'),
    ]),

    # Khu vực hiển thị kết quả
    html.Div(id='tabs-content', style={'padding': '20px'})
])

@callback(
    Output('tabs-content', 'children'),
    [Input('algo-tabs', 'value'), Input('upload-image', 'contents'), Input('k-slider', 'value')]
)
def render_content(tab, contents, k):
    if contents is None:
        return html.Div("Vui lòng upload ảnh để bắt đầu.", style={'textAlign': 'center', 'marginTop': '20px'})

    # Chọn thuật toán dựa trên Tab
    if tab == 'tab-svd':
        processed_img, retention, cum_var, size = process_svd(contents, k)
        algo_name = "Singular Value Decomposition (SVD)"
        color = 'blue'
    else:
        processed_img, retention, cum_var, size = process_pca(contents, k)
        algo_name = "Principal Component Analysis (PCA)"
        color = 'green'

    # Vẽ biểu đồ
    fig = go.Figure()
    # Nếu là PCA, cum_var có độ dài = k, nếu SVD thì độ dài = toàn bộ rank
    # Cần xử lý để vẽ cho đúng
    x_axis = list(range(1, len(cum_var) + 1))
    fig.add_trace(go.Scatter(x=x_axis, y=cum_var*100, mode='lines+markers', line=dict(color=color)))
    fig.update_layout(title="Mức độ giữ lại thông tin (%)", height=300, margin=dict(l=20, r=20, t=40, b=20))

    return html.Div([
        html.H3(f"Thuật toán: {algo_name}", style={'textAlign': 'center', 'color': color}),
        
        # Hàng hiển thị ảnh
        html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
            html.Div([html.H4("Ảnh Gốc"), html.Img(src=contents, style={'maxHeight': '300px'})]),
            html.Div([html.H4(f"Tái tạo (k={k})"), html.Img(src=processed_img, style={'maxHeight': '300px'})])
        ]),

        # Hàng hiển thị thông số
        html.Div(style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}, children=[
            html.P(f"Thông tin giữ lại: {retention:.2f}%"),
            html.P(f"Dung lượng nén ước tính: {size}")
        ]),

        # Biểu đồ
        dcc.Graph(figure=fig)
    ])

if __name__ == '__main__':
    app.run(debug=True)
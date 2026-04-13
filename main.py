from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = FastAPI()
app.mount("/Static", StaticFiles(directory="Static"), name="static")

templates = Jinja2Templates(directory="Templates")

@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse('Base.html', {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to prevent repeating code
def get_dataset_by_id(dataset_id: str):
    dataset_map = {
        "1": "Datasets/Vehicle_Telematics.csv",
        "2": "Datasets/Multi_Vehicle_Data.csv",
        "3": "Datasets/Driver_Behaviour.csv"
    }
    file_name = dataset_map.get(dataset_id)
    if not file_name:
        return None
    return load_and_clean(file_name)

def load_and_clean(dataset_name: str):
    try:
        data = pd.read_csv(dataset_name, low_memory=False)
        
        data = data.replace(',', '.', regex=True)
        data = data.replace(r'\.(?=\d{3})', '', regex=True)

        for col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
                errors='coerce'
            )

        data = data.dropna(axis=1, how='all')
        
        data = data.fillna(data.mean()).fillna(0) 

        return data
    except Exception as e:
        print(f"Loading Error: {e}")
        return pd.DataFrame() 

@app.get("/analyze/{dataset_id}")
async def run_lstm_analysis(dataset_id: str):
    df = get_dataset_by_id(dataset_id)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []
    for i in range(1, len(scaled_data)):
        X.append(scaled_data[i-1])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(X.shape[2])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0) 

    prediction = model.predict(X)
    error = np.nan_to_num(np.mean(np.abs(prediction - y), axis=1))
    threshold = np.mean(error) + 1 * np.std(error)
    sensistivity = 750
    health_score = round(max(0, 100 - (np.mean(error) * sensistivity)), 2)
    fault_events = int(np.sum(error > threshold))

    if health_score >= 90:
        condition = "Excellent, No maintenance required for a long period."
    elif health_score >= 75:
        condition = "Good, Minor wear may occur. Regular servicing recommended."
    elif health_score >= 50:
        condition = "Moderate, Maintenance may be required soon."
    else:
        condition = "Poor, Immediate inspection recommended."

    future_steps = 50
    last_input = X[-1]
    future_predictions = []
    current_input = last_input
    
    for _ in range(future_steps):
        pred = model.predict(current_input.reshape(1,1,X.shape[2]), verbose=0)
        future_predictions.append(pred[0])  
        current_input = pred
    
    future_predictions = np.array(future_predictions)
    future_predictions_orig = scaler.inverse_transform(np.array(future_predictions))
    
    y_true_scaled = y[:future_steps]
    future_error_scaled = np.mean(np.abs(future_predictions - y_true_scaled), axis=1)

    fig_future_error = px.line(y=future_error_scaled, title="Predicted Future Vehicle Error Trend", color_discrete_sequence= ['#E40046'])
    fig_future_error.update_layout(
        xaxis_title="Future Time Step",
        yaxis_title="Prediction Error",
        template="plotly_white",
        title_x=0.5,
        title_font=dict(size=22, color='#E40046')
    )

    base_error = np.mean(error)
    future_health = [max(0, 100 - (base_error * (1 + (i * 0.015)) * 100)) for i in range(future_steps)]
    
    fig_health_decline = px.line(
        y=future_health, 
        title="Predicted Vehicle Health Decline", 
        color_discrete_sequence=['#E40046']
    )

    fig_health_decline.update_layout(
        xaxis_title='Future Time Step',
        yaxis_title='Health Score',
        template='plotly_white', 
        title_x=0.5,
        title_font=dict(size=22, color='#E40046'),
        yaxis_range=[0, 105] 
    )
    
    return {
        "status": "success",
        "health_score": round(health_score, 2),
        "condition": condition,
        "fault_events": fault_events,
        "future_error_graph": json.dumps(fig_future_error, cls=plotly.utils.PlotlyJSONEncoder),
        "health_decline_graph": json.dumps(fig_health_decline, cls=plotly.utils.PlotlyJSONEncoder)
    }

@app.get("/visualize/{dataset_id}")
async def get_sensor_comparison(dataset_id: str, sensor1: str, sensor2: str):
    df = get_dataset_by_id(dataset_id)

    correlation = df[sensor1].corr(df[sensor2])
    corr_text = f"{correlation:.3f}" if not np.isnan(correlation) else "Constant values"

    fig = px.line(df, y=[sensor1, sensor2],  title=f"{sensor1} vs {sensor2}", labels={"value": sensor1, "index": sensor2}, color_discrete_sequence= ['#E40046', "gold"])
    
    fig.update_layout(yaxis=dict(autorange=True), template="plotly_white")
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return {"status": "success", "correlation": corr_text, "graph": graph_json}

@app.get("/get-sensors/{dataset_id}")
async def get_sensor_list(dataset_id: str):
    df = get_dataset_by_id(dataset_id)
    return {"sensors": list(df.columns)}

def style_graph(fig, title, x_label, y_label):
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': '#E40046', 'family': "Arial, sans-serif"}
        },
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#444")
    )
    fig.update_yaxes(exponentformat="none", showgrid=True, gridcolor='#eee')
    fig.update_xaxes(showgrid=False)
    return fig

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
# Smart Vehicle Health Monitoring Using OBD - II

Overview
-
This is an Smart Analysis Tool designed to predict vehicle health issues before they lead to critical failures. It helps in identifying problems at an early stage so that necessary actions can be taken in time. Unlike basic approaches that only detect faults after they occur, the SVHM system focuses on predictive maintenance. It works by identifying early signs of degradation in vehicle performance, allowing users to prevent failures instead of reacting to them later.

By implementing Long Short-Term Memory (LSTM) neural networks, this system continuously analyzes historical vehicle data obtained from an OBD-II (On-Board Diagnostics II) dataset sourced from Kaggle. OBD-II is a system in vehicles that monitors engine performance and other important components, collecting data from various sensors to detect faults or issues. This data can be accessed using a diagnostic tool, making it useful for analyzing vehicle health and performance. The system processes patterns and trends within this data to better understand vehicle behavior over time, enabling more accurate and reliable predictions related to vehicle health and performance.

LANGUAGES & TOOLS
-
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0B1C2C?style=for-the-badge&logo=python&logoColor=white)

Features 
-
-  Predicts potential vehicle errors.
- Cleans and analyzes historical vehicle telematics.
- Threshold Alerting | Anomaly Detection | Interactive Visualization.
- User-Centric, One-Click Diagnostic Reports.

Dataset Architecture
- 
1 Vehicle Telematics 
- Scale: ~1700 records| 30 Parameters.
- Focus: Core engine metrics.

2 Vehicle Batch Diagnostics
- Scale: ~ 60,0000 records | 14 Vehicles.
- Focus: Mixed models of cars (2003 - 2016).

3 Driver Behaviour Study
- Scale: ~8,200 records | 20 Drivers.
- Focus: Logs including Timing Advance and Fuel trims.
 

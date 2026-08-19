# 🚕 Chicago Urban Mobility Intelligence Platform
A production-ready geospatial analytics platform that combines satellite imagery, machine learning, residual diagnostics, and an LLM-powered spatial analyst for urban mobility forecasting in Chicago.

The system integrates multi-source spatial data, static and temporal XGBoost models, interactive mapping, and a tool-using conversational agent into a single Streamlit application.

Access the app here 👉:  https://bit.ly/4wNJQx1  
Watch usage video: https://youtu.be/VqZ9-ZvSHDY

---
Final platform view

<p align="center">
  <img src="https://github.com/frankraDIUM/Chicago-Urban-Mobility-Intelligence-Platform/blob/main/ch_mobi_ai.gif" />
</p>


---
Dashboard Preview

<p align="center">
  <img src="https://github.com/frankraDIUM/Chicago-Urban-Mobility-Pattern-Analysis-Using-Satellite-Imagery-Taxi-Data-and-ML/blob/main/mobility_analysis_.gif" />
</p>

---



## Overview

**Chicago Urban Mobility Intelligence** is an end-to-end decision-support platform for exploring and forecasting taxi demand across Chicago’s 77 community areas.

It brings together:
- Sentinel-2 satellite-derived vehicle density proxies
- 14M+ Chicago taxi trip records
- OpenStreetMap road network and business/POI data
- Population density (WorldPop)
- Static (community-area) and temporal (hourly) XGBoost models
- Residual analysis for hidden demand hotspots
- An LLM-powered AI Mobility Analyst with tool calling

The platform is designed for both research insight and operational exploration (maps, forecasts, and conversational spatial analysis).

---

## Dashboard

The main **Dashboard** tab provides:
- Interactive Folium maps with multiple layers (Total Taxi Trips, Trips per 1,000 People, Satellite Proxy, Residuals)
- Custom basemap options (CartoDB Positron, Satellite, OpenStreetMap)
- Quick Temporal Forecast panel (select community area, hour, and weekend flag)
- Automatic map highlight and zoom to the predicted community area
- Live prediction metric display

Users can explore spatial demand patterns and generate hourly forecasts in one interface.

---

## AI Mobility Analyst

The **AI Mobility Analyst** tab pairs an interactive map with a tool-using LLM agent (Groq / Llama 3.3).

The agent can:
- Predict hourly taxi demand for any community area
- Retrieve highest-demand neighborhoods
- Report spatial model feature importance
- Identify hidden hotspots (positive residuals)
- Return detailed statistics for a selected area
- Automatically update map layer and focus based on the query

Responses are analytical (not raw tool dumps) and oriented toward transportation planning insight.

---

## Methodology

### Data Sources
- **Satellite**: Sentinel-2 MSI (10 m bands B02, B03, B04, B08) for texture / vehicle density proxies
- **Mobility**: Chicago Taxi Trips (~14M records) as ground truth
- **Roads**: OpenStreetMap network extract
- **Population**: WorldPop 2020 gridded estimates
- **POI / Business**: Chicago Business Licenses (hotels, bars, restaurants, etc.)

### Modeling Pipeline
1. Spatial feature engineering (POI counts, airport flag, distance to Loop, satellite proxy, population density)
2. Static XGBoost model → annual demand per community area
3. Residual analysis → hidden hotspots and over-prediction zones
4. Temporal aggregation (hour × day type × community area)
5. Temporal XGBoost model (native DMatrix) → hourly demand forecasts
6. Interactive Streamlit interface + LLM agent with tool calling

---

## Results

### Model Performance
| Model              | Metric     | Value    |
|--------------------|------------|----------|
| Static (community) | R²         | 0.666    |
| Static             | MAE        | 50,806 trips |
| Temporal (hourly)  | R²         | **0.954** |
| Temporal           | MAE        | **294 trips** |

### Key Drivers (Static Model)
1. **num_hotels** — 0.487  
2. **is_airport** — 0.217  
3. **dist_to_loop_km** — 0.082  
4. **num_bars** — 0.062  
5. **num_restaurants** — 0.055  

**Main findings**
- Taxi demand is strongly destination-driven (hotels and airports dominate).
- Sentinel-2 10 m texture proxy adds limited predictive value once POI and location features are included.
- O’Hare and Near North Side remain significant under-predicted (hidden) hotspots.
- Several far South Side areas are systematically over-predicted.

---


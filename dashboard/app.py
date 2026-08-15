# Chicago Urban Mobility Intelligence
import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
import numpy as np
import joblib
import xgboost as xgb
from groq import Groq
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

LLM_MODEL = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="Chicago Mobility Intelligence",
    layout="wide",
    page_icon="🚕"
)

# ----------------------------------------------------------------------
# Enterprise UI Theme & Custom Styling (CSS)
# ----------------------------------------------------------------------
st.markdown("""
<style>
/* Hide/unblock Streamlit's default top header bar that overlaps content */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1rem !important;
}

/* Force container down with ample clear margin */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem !important;
}

/* Ensure clean title line-height without top clipping */
h1 {
    color: #1565C0 !important;
    font-weight: 700 !important;
    margin-top: 0rem !important;
    padding-top: 0rem !important;
    line-height: 1.3 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D0D7DE;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background-color: #E9EEF5;
}

/* Custom UI Cards for Data Sources & Architecture */
.data-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    height: 100%;
}
.data-card h5 {
    color: #0F172A;
    margin-bottom: 6px;
    font-weight: 600;
}
.data-card p {
    color: #475569;
    font-size: 0.88rem;
    margin: 0;
}

/* Architecture / Workflow Box */
.workflow-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 16px;
    font-family: monospace;
    font-size: 0.88rem;
    color: #1E293B;
    text-align: center;
    line-height: 1.6;
}

/* Platform Overview Banner */
.overview-banner {
    background: linear-gradient(135deg, #0D47A1 0%, #1976D2 100%);
    color: #FFFFFF;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(13, 71, 161, 0.15);
}
.overview-banner p {
    color: #E3F2FD;
    font-size: 1.02rem;
    margin: 0;
    line-height: 1.5;
}

/* Capability Badge List */
.capability-item {
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
}

/* Sidebar System Status Styling */
.status-card {
    background-color: #FFFFFF;
    border: 1px solid #D0D7DE;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
}
.status-item {
    font-size: 0.85rem;
    margin-bottom: 4px;
    font-weight: 500;
}
.status-ok {
    color: #166534;
}
.status-err {
    color: #991B1B;
}

/* Footer Styling */
.app-footer {
    border-top: 1px solid #E2E8F0;
    padding-top: 16px;
    margin-top: 40px;
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='font-size:1.75rem; margin-bottom:0.6rem;'>🚕 Chicago Urban Mobility Intelligence</h1>",
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------
# Data & Model Loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        geojson_path = BASE_DIR / "chicago_community_areas_with_residuals.geojson"
        comm = gpd.read_file(geojson_path)

        for col in comm.columns:
            if pd.api.types.is_datetime64_any_dtype(comm[col]):
                comm[col] = comm[col].astype(str)

        if 'trips_per_1000_people' in comm.columns:
            comm['trips_per_1k_fmt'] = comm['trips_per_1000_people'].round(1)
        if 'total_trips' in comm.columns:
            comm['total_trips_fmt'] = comm['total_trips'].fillna(0).astype(int)
        if 'residual' in comm.columns:
            comm['residual_fmt'] = comm['residual'].fillna(0).astype(int)
        if 'avg_satellite_proxy' in comm.columns:
            comm['satellite_fmt'] = comm['avg_satellite_proxy'].round(2)

        data_df = comm.drop(columns=['geometry']).copy()
        return comm, data_df
    except Exception as e:
        st.error(f"Could not load spatial dataset (`chicago_community_areas_with_residuals.geojson`): {e}")
        return gpd.GeoDataFrame(), pd.DataFrame()

comm_areas, data_df = load_data()
geo_json = json.loads(comm_areas.to_json()) if not comm_areas.empty else {}

@st.cache_resource
def load_temporal_model():
    try:
        model_path = BASE_DIR / "taxi_temporal_model_native.pkl"
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load temporal XGBoost model (`taxi_temporal_model_native.pkl`): {e}")
        return None

temporal_model = load_temporal_model()

area_col = 'community' if 'community' in comm_areas.columns else 'pickup_community_area'
area_list = sorted(comm_areas[area_col].unique().tolist()) if not comm_areas.empty else []

LAYER_OPTIONS = ["Total Taxi Trips", "Trips per 1,000 People", "Satellite Proxy", "Residuals", "None"]

# ----------------------------------------------------------------------
# Session State
# ----------------------------------------------------------------------
if "selected_area" not in st.session_state:
    st.session_state.selected_area = None
if "map_layer" not in st.session_state:
    st.session_state.map_layer = "Total Taxi Trips"
if "basemap" not in st.session_state:
    st.session_state.basemap = "CartoDB positron"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool" not in st.session_state:
    st.session_state.pending_tool = None
if "map_version" not in st.session_state:
    st.session_state.map_version = 0
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# ----------------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------------
def predict_trips(area: str, hour: int, weekend: bool) -> int:
    if temporal_model is None or comm_areas.empty:
        return -1
    try:
        area_id = comm_areas[comm_areas[area_col] == area]['pickup_community_area'].iloc[0]
        input_df = pd.DataFrame({
            'pickup_community_area': [area_id],
            'hour': [hour],
            'is_weekend': [1 if weekend else 0],
            'hour_sin': [np.sin(2 * np.pi * hour / 24)],
            'hour_cos': [np.cos(2 * np.pi * hour / 24)]
        })
        pred = temporal_model.predict(xgb.DMatrix(input_df))[0]
        return int(max(0, pred))
    except Exception:
        return -1

def get_highest_demand_areas(n: int = 5) -> str:
    if data_df.empty:
        return "Dataset unavailable."
    top = data_df.nlargest(n, 'total_trips')[[area_col, 'total_trips', 'trips_per_1000_people']]
    lines = [
        f"{i+1}. {row[area_col]} — {int(row['total_trips']):,} trips "
        f"({row['trips_per_1000_people']:.1f} per 1k people)"
        for i, (_, row) in enumerate(top.iterrows())
    ]
    return "Highest demand areas:\n" + "\n".join(lines)

def get_feature_importance() -> str:
    return (
        "Top feature importances from the spatial model:\n"
        "1. num_hotels          → 0.487\n"
        "2. is_airport          → 0.217\n"
        "3. dist_to_loop_km     → 0.082\n"
        "4. num_bars            → 0.062\n"
        "5. num_restaurants     → 0.055\n"
        "Hotels and airport presence dominate predicted demand."
    )

def find_hidden_hotspots(n: int = 5) -> str:
    if data_df.empty:
        return "Dataset unavailable."
    hot = data_df.nlargest(n, 'residual')[[area_col, 'residual', 'total_trips']]
    lines = [
        f"{i+1}. {row[area_col]} — residual +{int(row['residual']):,} "
        f"(observed trips: {int(row['total_trips']):,})"
        for i, (_, row) in enumerate(hot.iterrows())
    ]
    return "Hidden hotspots (model under-predicts):\n" + "\n".join(lines)

def get_area_stats(area: str) -> str:
    if data_df.empty:
        return "Dataset unavailable."
    row = data_df[data_df[area_col] == area]
    if row.empty:
        return f"Area '{area}' not found."
    r = row.iloc[0]
    return (
        f"Stats for {area}:\n"
        f"- Total trips: {int(r['total_trips']):,}\n"
        f"- Trips per 1,000 people: {r['trips_per_1000_people']:.1f}\n"
        f"- Residual: {int(r['residual']):,}\n"
        f"- Satellite proxy: {r.get('avg_satellite_proxy', 'N/A')}"
    )

# ----------------------------------------------------------------------
# Groq Client & Tools
# ----------------------------------------------------------------------
@st.cache_resource
def get_groq_client():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return Groq(api_key=st.secrets["GROQ_API_KEY"])
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Groq Client Error: {e}")
        return None

client = get_groq_client()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_trips",
            "description": "Predict taxi trips for a community area at a given hour and weekend flag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "description": f"Exact name from: {', '.join(area_list)}"},
                    "hour": {"type": "integer", "description": "0-23"},
                    "weekend": {"type": "boolean"}
                },
                "required": ["area", "hour", "weekend"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_highest_demand_areas",
            "description": "Return top-n highest demand community areas.",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_importance",
            "description": "Return the strongest predictors in the spatial model.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_hidden_hotspots",
            "description": "Return areas where the model under-predicts (positive residuals).",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_area_stats",
            "description": "Detailed statistics for one community area.",
            "parameters": {
                "type": "object",
                "properties": {"area": {"type": "string"}},
                "required": ["area"]
            }
        }
    }
]

def run_tool(name: str, args: dict) -> str:
    if name == "predict_trips":
        pred = predict_trips(args.get("area", ""), int(args.get("hour", 12)), bool(args.get("weekend", False)))
        if pred < 0:
            return "Prediction failed."
        day = "weekend" if args.get("weekend") else "weekday"
        return f"Predicted taxi trips for {args.get('area')} at hour {args.get('hour')} ({day}): {pred:,}"
    elif name == "get_highest_demand_areas":
        return get_highest_demand_areas(int(args.get("n", 5)))
    elif name == "get_feature_importance":
        return get_feature_importance()
    elif name == "find_hidden_hotspots":
        return find_hidden_hotspots(int(args.get("n", 5)))
    elif name == "get_area_stats":
        return get_area_stats(args.get("area", ""))
    return "Unknown tool."

def clean_answer(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<function=.*?</function>', '', text, flags=re.DOTALL)
    text = re.sub(r'<function=.*?>', '', text)
    text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'Tool call:.*?(\n|$)', '', text)
    return text.strip()

# ----------------------------------------------------------------------
# Custom Legend
# ----------------------------------------------------------------------
def add_custom_legend(m, title, bins, colors):
    legend_html = f'''
     <div style="position: fixed;
     bottom: 30px; left: 30px; width: auto; min-width: 170px; height: auto;
     background-color: white; border:2px solid #555; z-index:9999; font-size:13px;
     padding: 12px; color: black; box-shadow: 2px 2px 6px rgba(0,0,0,0.25); border-radius: 6px;">
     <b style="display: block; margin-bottom: 8px; font-size: 13px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">{title}</b>
     <table style="border-spacing: 0 4px; border-collapse: separate;">
    '''
    for i in range(len(colors)):
        start = f"{bins[i]:,.1f}" if "Trips / 1k" in title else f"{int(bins[i]):,}"
        end = f"{bins[i+1]:,.1f}" if "Trips / 1k" in title else f"{int(bins[i+1]):,}"
        legend_html += f'''
        <tr>
            <td style="vertical-align: middle;">
                <div style="background:{colors[i]}; width:18px; height:18px; border:1px solid #333;"></div>
            </td>
            <td style="vertical-align: middle; padding-left: 8px; white-space: nowrap; font-weight: 500;">
                {start} &ndash; {end}
            </td>
        </tr>
        '''
    legend_html += '</table></div>'
    m.get_root().html.add_child(folium.Element(legend_html))

# ----------------------------------------------------------------------
# Map Builder
# ----------------------------------------------------------------------
def build_map(layer: str, basemap: str, highlight_area: str | None = None):
    if basemap == "Satellite":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri"
    elif basemap == "OpenStreetMap":
        tiles = "OpenStreetMap"
        attr = "OpenStreetMap"
    else:
        tiles = "CartoDB positron"
        attr = "CartoDB"

    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles=tiles, attr=attr)

    if layer != "None" and not comm_areas.empty:
        if layer == "Satellite Proxy":
            folium.Choropleth(
                geo_data=geo_json, data=data_df,
                columns=['pickup_community_area', 'satellite_fmt'],
                key_on='feature.properties.pickup_community_area',
                fill_color='YlOrRd', fill_opacity=0.75, line_opacity=0.3,
                legend_name='Satellite Proxy'
            ).add_to(m)
        else:
            if layer == "Total Taxi Trips":
                col_target, col_tooltip, title = 'total_trips', 'total_trips_fmt', "Total Taxi Trips"
                tooltip_alias = "Total Trips"
                colors = ['#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
                bins = np.linspace(data_df[col_target].min(), data_df[col_target].max() * 1.0001, 7)
            elif layer == "Trips per 1,000 People":
                col_target, col_tooltip, title = 'trips_per_1000_people', 'trips_per_1k_fmt', "Trips / 1k People"
                tooltip_alias = "Trips / 1k People"
                colors = ['#f2f0f7', '#dadaeb', '#bcbddc', '#9e9ac8', '#756bb1', '#54278f']
                bins = np.linspace(data_df[col_target].min(), data_df[col_target].max() * 1.0001, 7)
            else:  # Residuals
                col_target, col_tooltip, title = 'residual', 'residual_fmt', "Residuals"
                tooltip_alias = "Residual"
                colors = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020']
                max_val = data_df[col_target].abs().max()
                bins = np.linspace(-max_val, max_val * 1.0001, 6)

            def get_color(val):
                if val is None or np.isnan(val):
                    return '#808080'
                for i in range(len(bins) - 1):
                    if bins[i] <= val < bins[i + 1]:
                        return colors[i]
                return colors[-1]

            def style_function(feature):
                props = feature["properties"]
                name = str(props.get(area_col) or props.get("pickup_community_area", "")).strip().lower()
                target = (highlight_area or "").strip().lower()

                base_fill = get_color(props.get(col_target, 0))

                if target and name == target:
                    return {
                        "fillColor": base_fill,
                        "color": "#00E5FF",          # cyan border highlight
                        "weight": 4.5,
                        "fillOpacity": 0.85
                    }
                return {
                    "fillColor": base_fill,
                    "color": "#333333",
                    "weight": 0.6,
                    "fillOpacity": 0.72
                }

            folium.GeoJson(
                comm_areas,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=[area_col, col_tooltip] if area_col in comm_areas.columns
                    else ["pickup_community_area", col_tooltip],
                    aliases=["Community Area:", f"{tooltip_alias}:"],
                    localize=True
                )
            ).add_to(m)

            add_custom_legend(m, title, bins, colors)

    if highlight_area and not comm_areas.empty:
        try:
            geom = comm_areas[comm_areas[area_col].str.lower() == highlight_area.strip().lower()]
            if not geom.empty:
                bounds = geom.total_bounds  # [minx, miny, maxx, maxy]
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], padding=(50, 50), max_zoom=13)
        except Exception:
            pass

    return m

# ----------------------------------------------------------------------
# Sidebar System Status Panel
# ----------------------------------------------------------------------
st.sidebar.markdown("### System Status")

status_data = "✓ Spatial dataset loaded" if not comm_areas.empty else " Missing spatial GeoJSON"
status_model = "✓ Temporal XGBoost loaded" if temporal_model is not None else " Temporal model offline"
status_ai = (
            f"✓ AI Analyst connected ({LLM_MODEL})"
            if client is not None
            else " Groq API key missing"
        )
status_map = "✓ Map engine ready"

st.sidebar.markdown(f"""
<div class="status-card">
    <div class="status-item {'status-ok' if not comm_areas.empty else 'status-err'}">{status_data}</div>
    <div class="status-item {'status-ok' if temporal_model is not None else 'status-err'}">{status_model}</div>
    <div class="status-item {'status-ok' if client is not None else 'status-err'}">{status_ai}</div>
    <div class="status-item status-ok">{status_map}</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Navigation Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Dashboard",
    "🤖 AI Mobility Analyst",
    "📈 Analytics",
    "⚙ About the Model"
])

# ===================== TAB 1: Dashboard =====================
with tab1:
    st.subheader("Operational Dashboard")

    ctrl_left, ctrl_right, _ = st.columns([1.1, 1.3, 1.6])
    with ctrl_left:
        layer = st.selectbox(
            "Map Layer",
            LAYER_OPTIONS,
            index=LAYER_OPTIONS.index(st.session_state.map_layer) if st.session_state.map_layer in LAYER_OPTIONS else 0
        )
        st.session_state.map_layer = layer
    with ctrl_right:
        basemap = st.radio(
            "Base Map",
            ["CartoDB positron", "Satellite", "OpenStreetMap"],
            horizontal=True,
            index=["CartoDB positron", "Satellite", "OpenStreetMap"].index(st.session_state.basemap)
            if st.session_state.basemap in ["CartoDB positron", "Satellite", "OpenStreetMap"] else 0
        )
        st.session_state.basemap = basemap

    map_col, spacer, panel_col = st.columns([2.05, 0.08, 0.95])

    with panel_col:
        st.markdown("#### Quick Temporal Forecast")

        sel_area = st.selectbox("Community Area", area_list, key="dash_area")
        sel_hour = st.slider("Hour of Day", 0, 23, 17, key="dash_hour")
        is_wknd = st.checkbox("Weekend?", key="dash_wknd")

        btn_col, _ = st.columns([1.2, 1])
        with btn_col:
            run_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

        if run_clicked:
            pred = predict_trips(sel_area, sel_hour, is_wknd)
            if pred >= 0:
                st.session_state.latest_prediction = {
                    "area": sel_area,
                    "hour": sel_hour,
                    "weekend": is_wknd,
                    "trips": pred
                }
                st.session_state.selected_area = sel_area
                st.session_state.map_version += 1          # force map refresh
                st.rerun()
            else:
                st.error("Prediction failed")

        if st.session_state.latest_prediction:
            p = st.session_state.latest_prediction
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(
                label=f"Predicted Trips ({p['area']})",
                value=f"{p['trips']:,}",
                delta=f"Hour {p['hour']}:00 | {'Weekend' if p['weekend'] else 'Weekday'}"
            )

        st.markdown("---")
        st.caption("The map highlights and zooms to the predicted community.")

    with map_col:
        m = build_map(st.session_state.map_layer, st.session_state.basemap, st.session_state.selected_area)
        map_key = f"dash_map_{st.session_state.selected_area}_{st.session_state.map_version}"
        st_folium(m, use_container_width=True, height=720, returned_objects=[], key=map_key)

# ===================== TAB 2: AI Mobility Analyst =====================
with tab2:
    st.subheader("AI Geospatial Analyst")

    map_col, spacer, chat_col = st.columns([2.05, 0.08, 0.95])

    with map_col:
        mc1, mc2 = st.columns(2)
        with mc1:
            layer_ai = st.selectbox(
                "Layer",
                LAYER_OPTIONS,
                index=LAYER_OPTIONS.index(st.session_state.map_layer) if st.session_state.map_layer in LAYER_OPTIONS else 0,
                key="ai_layer"
            )
            st.session_state.map_layer = layer_ai
        with mc2:
            basemap_ai = st.selectbox(
                "Basemap",
                ["CartoDB positron", "Satellite", "OpenStreetMap"],
                index=["CartoDB positron", "Satellite", "OpenStreetMap"].index(st.session_state.basemap)
                if st.session_state.basemap in ["CartoDB positron", "Satellite", "OpenStreetMap"] else 0,
                key="ai_basemap"
            )
            st.session_state.basemap = basemap_ai

        m_ai = build_map(st.session_state.map_layer, st.session_state.basemap, st.session_state.selected_area)
        ai_map_key = f"ai_map_{st.session_state.selected_area}_{st.session_state.map_version}"
        st_folium(m_ai, use_container_width=True, height=720, returned_objects=[], key=ai_map_key)

    with chat_col:
        st.markdown("##### 🤖 Mobility Assistant")

        if client is None:
            st.error("Add `GROQ_API_KEY` to `.streamlit/secrets.toml`")
        else:
            SYSTEM_PROMPT = {
                "role": "system",
                "content": """
You are an urban mobility intelligence analyst for Chicago taxi demand.

Rules you must follow:
- Never simply repeat tool outputs.
- Always produce an analytical explanation that a transportation planner would find useful.
- When a prediction or statistic is returned:
  1. Explain what the number means in practical terms.
  2. Relate it to spatial characteristics (hotels, airport, distance to Loop, commercial density).
  3. Reference the known strongest drivers (hotels 0.487, airport 0.217, distance to Loop, bars, restaurants) when relevant.
  4. Compare to other parts of the city when useful.
  5. Mention residual status if available (hidden hotspot vs well-captured).
- You may call multiple tools in one turn when it improves the analysis.
- After tools have returned results, your final reply must be pure natural language.
- NEVER output function-call syntax, XML tags, or tool names in the final answer.
"""
            }

            if not st.session_state.messages:
                st.session_state.messages = [SYSTEM_PROMPT]

            # Fixed height scrollable container for chat history
            chat_container = st.container(height=600)

            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] in ("user", "assistant"):
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                if st.session_state.pending_tool:
                    pending = st.session_state.pending_tool
                    st.info(f"Run **{pending['friendly_name']}** for **{pending.get('area', '')}**?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, run it", key="confirm_yes"):
                            result = run_tool(pending["name"], pending["args"])
                            st.session_state.messages.append({"role": "assistant", "content": f"(Running {pending['friendly_name']})"})
                            st.session_state.messages.append({"role": "tool", "name": pending["name"], "content": result, "tool_call_id": "manual"})

                            final = client.chat.completions.create(
                                model=LLM_MODEL,
                                messages=st.session_state.messages + [{"role": "user", "content": "Now give a short analytical explanation of the new result."}],
                                temperature=0.3
                            )
                            answer = clean_answer(final.choices[0].message.content)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.session_state.pending_tool = None
                            st.rerun()
                    with col_no:
                        if st.button("No, thanks", key="confirm_no"):
                            st.session_state.pending_tool = None
                            st.rerun()

            # Pinned input box beneath viewport
            if prompt := st.chat_input("Ask a mobility question…"):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.status("Analyzing spatial query...", expanded=True) as status:
                        try:
                            status.write("Evaluating query & selecting appropriate spatial tools...")
                            response = client.chat.completions.create(
                                model=LLM_MODEL,
                                messages=st.session_state.messages,
                                tools=TOOLS,
                                tool_choice="auto",
                                temperature=0.3
                            )
                            msg = response.choices[0].message

                            if msg.tool_calls:
                                tool_results = []
                                for tool_call in msg.tool_calls:
                                    fn_name = tool_call.function.name
                                    try:
                                        fn_args = json.loads(tool_call.function.arguments)
                                    except Exception:
                                        fn_args = {}

                                    status.write(f"Executing tool: `{fn_name}`...")
                                    result = run_tool(fn_name, fn_args)
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": fn_name,
                                        "content": result
                                    })

                                    if fn_name in ("predict_trips", "get_area_stats") and "area" in fn_args:
                                        st.session_state.selected_area = fn_args["area"]
                                        st.session_state.map_version += 1
                                    if fn_name == "find_hidden_hotspots":
                                        st.session_state.map_layer = "Residuals"
                                    if fn_name == "get_highest_demand_areas":
                                        st.session_state.map_layer = "Total Taxi Trips"

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": msg.content or "",
                                    "tool_calls": [
                                        {"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls
                                    ]
                                })
                                for tr in tool_results:
                                    st.session_state.messages.append(tr)

                                status.write("Synthesizing analytical findings...")
                                final = client.chat.completions.create(
                                    model=LLM_MODEL,
                                    messages=st.session_state.messages + [{
                                        "role": "system",
                                        "content": "Respond with pure natural language only. Do not emit any function calls, XML, or tool syntax."
                                    }],
                                    temperature=0.3
                                )
                                answer = clean_answer(final.choices[0].message.content)
                                status.update(label="Analysis complete!", state="complete", expanded=False)

                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.rerun()

                            else:
                                answer = clean_answer(msg.content)
                                status.update(label="Response ready!", state="complete", expanded=False)
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                                st.rerun()

                        except Exception as e:
                            status.update(label="Error processing query", state="error")
                            st.error(f"Agent error: {e}")

# ===================== TAB 3: Analytics =====================
with tab3:
    st.subheader("Spatial & Temporal Analytics")

    # Section 1: Model Performance Highlights
    st.markdown("#### Model Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Static Model R²", "0.666", help="Variance explained across 77 community areas")
    p2.metric("Static Model MAE", "50,806", delta="Trips / Year", delta_color="off")
    p3.metric("Temporal Model R²", "0.954", help="Hourly prediction accuracy")
    p4.metric("Temporal Model MAE", "294", delta="Trips / Hour", delta_color="off")

    st.markdown("---")

    # Section 2: Demand Drivers
    col_feat, col_text = st.columns([1.6, 1.0])
    with col_feat:
        st.markdown("#### Demand Drivers")
        imp_data = pd.DataFrame({
            'Feature': ['num_hotels', 'is_airport', 'dist_to_loop_km', 'num_bars', 'num_restaurants'],
            'Importance': [0.487, 0.217, 0.082, 0.062, 0.055]
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            imp_data, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues',
            text_auto='.3f', title="Feature Importance (XGBoost Static Model)"
        )
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        st.markdown("#### Driver Insights")
        st.markdown("""
        - **Hotel Density Dominance (48.7%):** Lodging infrastructure is by far the strongest predictor of urban taxi originations.
        - **Airport Infrastructure (21.7%):** Dedicated transport hubs (O'Hare and Midway) generate distinct, high-volume travel demand independent of population.
        - **Spatial Centrality (8.2%):** Euclidean distance to the Loop captures core urban density gradients across outer neighborhoods.
        """)

    st.markdown("---")

    # Section 3: Hidden Hotspots (Residuals)
    st.markdown("#### Hidden Hotspots (Residual Analysis)")
    col_map_res, col_tbl_res = st.columns([1.5, 1.1])

    with col_map_res:
        st.caption("Red = Model Under-predicts (Actual Demand > Predicted) | Blue = Model Over-predicts")
        m_res = build_map("Residuals", "CartoDB positron", None)
        st_folium(m_res, use_container_width=True, height=450, returned_objects=[], key="analytics_residuals_map")

    with col_tbl_res:
        st.markdown("##### Top Under-Predicted Neighborhoods")
        if not data_df.empty:
            top_hot = data_df.nlargest(6, 'residual')[[area_col, 'residual', 'total_trips']].copy()
            top_hot.columns = ['Community Area', 'Residual (+Trips)', 'Observed Trips']
            top_hot['Residual (+Trips)'] = top_hot['Residual (+Trips)'].map('{:,.0f}'.format)
            top_hot['Observed Trips'] = top_hot['Observed Trips'].map('{:,.0f}'.format)
            st.dataframe(top_hot, use_container_width=True, hide_index=True)
        st.info("Positive residuals identify 'hidden hotspots' where non-residential factors (e.g. seasonal events, transient venues) drive demand above structural baseline expectations.")

    st.markdown("---")

    # Section 4: Temporal Demand
    st.markdown("#### Temporal Demand Profiling")

    sel_col, _ = st.columns([1.2, 1])
    with sel_col:
        temp_area = st.selectbox(
            "Select Community Area for 24-Hour Profile",
            area_list,
            index=area_list.index("Near North Side") if "Near North Side" in area_list else 0,
            key="analytics_temp_area"
        )

    if temp_area:
        hours = list(range(24))
        weekday_preds = [predict_trips(temp_area, h, weekend=False) for h in hours]
        weekend_preds = [predict_trips(temp_area, h, weekend=True) for h in hours]

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=hours, y=weekday_preds, mode='lines+markers', name='Weekday', line=dict(color='#1565C0', width=3)))
        fig_temp.add_trace(go.Scatter(x=hours, y=weekend_preds, mode='lines+markers', name='Weekend', line=dict(color='#E65100', width=3, dash='dash')))
        fig_temp.update_layout(
            title=f"24-Hour Hourly Taxi Demand Profile — {temp_area}",
            xaxis_title="Hour of Day (0 - 23)",
            yaxis_title="Predicted Taxi Trips",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

# ===================== TAB 4: About the Model =====================
with tab4:
    # Overview Banner
    st.markdown("""
    <div class="overview-banner">
        <h3 style="margin-top:0; margin-bottom:8px; color:#FFFFFF;">Platform Overview</h3>
        <p><b>Chicago Urban Mobility Intelligence</b> is an interactive decision-support platform that combines machine learning, geospatial analytics, and a tool-using LLM agent to explore taxi demand patterns across Chicago. The system integrates spatial prediction, temporal forecasting, residual diagnostics, and conversational analytics into a single unified interface.</p>
    </div>
    """, unsafe_allow_html=True)

    # Modeling Pipeline Workflow Diagram
    st.markdown("#### System Architecture & Pipeline")
    st.markdown("""
    <div class="workflow-box">
    Sentinel-2 Satellite & POI Data &nbsp;──►&nbsp; Spatial Feature Engineering &nbsp;──►&nbsp; XGBoost Static Model &nbsp;──►&nbsp; Demand Diagnostics<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
    Groq GPT-OSS 120B LLM Agent &nbsp;&lt;──►&nbsp; Interactive Folium Map Dashboard &nbsp;&lt;──►&nbsp; Temporal Hourly Model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data Sources Cards
    st.markdown("#### Data Sources")
    ds1, ds2, ds3, ds4, ds5 = st.columns(5)

    with ds1:
        st.markdown("""
        <div class="data-card">
            <h5>📡 Satellite</h5>
            <p><b>Sentinel-2 MSI</b><br>10m resolution bands (B02, B03, B04, B08) for vehicle proxy texture analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with ds2:
        st.markdown("""
        <div class="data-card">
            <h5>🚕 Mobility</h5>
            <p><b>Chicago Taxi Trips</b><br>14M+ ground truth trips across 77 Chicago community areas.</p>
        </div>
        """, unsafe_allow_html=True)

    with ds3:
        st.markdown("""
        <div class="data-card">
            <h5>🛣️ Roads</h5>
            <p><b>OpenStreetMap</b><br>OSM network extract for spatial accessibility and road proximity.</p>
        </div>
        """, unsafe_allow_html=True)

    with ds4:
        st.markdown("""
        <div class="data-card">
            <h5>👥 Population</h5>
            <p><b>WorldPop 2020</b><br>Gridded 100m high-resolution population density estimates.</p>
        </div>
        """, unsafe_allow_html=True)

    with ds5:
        st.markdown("""
        <div class="data-card">
            <h5>🏨 POI / Business</h5>
            <p><b>Business Licenses</b><br>Chicago business dataset mapping hotel, bar, and venue density.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Architecture & Capabilities
    col_arch, col_caps = st.columns([1.2, 1.0])

    with col_arch:
        st.markdown("#### Model Architecture")
        st.markdown("""
        **Spatial Model (Static)**
        - **Algorithm:** XGBoost Regressor (`n_estimators=300`, `max_depth=5`)
        - **Input Features:** Hotel count, Restaurant count, Bar count, Airport flag, Distance to Loop (km), Population density, Satellite texture proxy.
        - **Output Target:** Annual cumulative originations per community area.

        **Temporal Model (Hourly)**
        - **Algorithm:** XGBoost Native DMatrix model
        - **Input Features:** `pickup_community_area`, `hour`, `is_weekend`, `hour_sin`, `hour_cos`
        - **Output Target:** Hourly taxi demand prediction per community area.
        """)

    with col_caps:
        st.markdown("#### AI Analyst Capabilities")
        st.markdown("""
        <div class="capability-item">✓ Predict hourly taxi demand dynamically</div>
        <div class="capability-item">✓ Explain spatial model feature behavior</div>
        <div class="capability-item">✓ Identify under-predicted hidden hotspots</div>
        <div class="capability-item">✓ Compare neighborhood mobility metrics</div>
        <div class="capability-item">✓ Automatically switch map layers & zoom focus</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Known Limitations
    st.markdown("#### Known Limitations")
    st.warning("""
    - **Spatial Aggregation:** Community-area spatial boundaries may mask fine-grained neighborhood micro-mobility patterns.
    - **Satellite Resolution:** 10m Sentinel-2 optical imagery cannot directly isolate individual moving vehicles in dense urban shadows.
    - **Mode Scope:** Dataset covers licensed taxi trips and excludes private ride-hailing services (e.g. Uber / Lyft).
    - **Temporal Scope:** Predictions reflect historical baseline spatial demand patterns rather than real-time traffic incidents.
    """)

# ----------------------------------------------------------------------
# Portfolio Footer
# ----------------------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <b>Chicago Urban Mobility Intelligence v1.0</b><br>
    Built by Frank G. Asiamah &nbsp;|&nbsp; Python · Streamlit · GeoPandas · XGBoost · Folium · GPT-OSS 120B · Groq
</div>
""", unsafe_allow_html=True)

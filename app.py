import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import time
import json
import os
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. PAGE SETUP & LOCAL DATA PATHS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise AI Travel Agent", layout="wide", page_icon="🗺️")
st.title("🗺️ Enterprise AI Travel Agent")
st.write("Production-grade autonomous travel planner equipped with a persistent learning feedback engine.")

FEEDBACK_FILE = "data/feedback.jsonl"
os.makedirs("data", exist_ok=True)

USER_AGENT = "trip-planner-capstone/1.0 (mycustomtestproject99@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# 2. STATE INTERACTION ENGINE (SESSION STATE)
# ---------------------------------------------------------------------------
if "itinerary_data" not in st.session_state:
    st.session_state.itinerary_data = None
if "geo_info" not in st.session_state:
    st.session_state.geo_info = None

# ---------------------------------------------------------------------------
# 3. CORE FEEDBACK DATA LAYER (MANAGEMENT & MATHEMATICAL SCORES)
# ---------------------------------------------------------------------------
def save_feedback(city_key, poi_id, vote):
    """Appends structural telemetry click events instantly into the local JSONL file."""
    feedback_event = {
        "ts": time.time(),
        "city_key": city_key.lower().strip(),
        "poi_id": poi_id,
        "vote": vote
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback_event) + "\n")

def calculate_poi_boosts(city_key):
    """
    Parses historical interactions to calculate real-time mathematical boosts:
    Upvotes yield +0.25 reinforcement weights. Downvotes yield -0.35 penalties.
    """
    boost_map = {}
    normalized_city = city_key.lower().strip()
    
    if not os.path.exists(FEEDBACK_FILE):
        return boost_map
        
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    event = json.loads(line)
                    if event.get("city_key") == normalized_city:
                        poi = event.get("poi_id")
                        vote = event.get("vote")
                        
                        current_boost = boost_map.get(poi, 0.0)
                        if vote == "up":
                            boost_map[poi] = current_boost + 0.25
                        elif vote == "down":
                            boost_map[poi] = current_boost - 0.35
                except Exception:
                    continue
    return boost_map

# ---------------------------------------------------------------------------
# 4. GEOGRAPHIC LOOKUP COMPLIANCE ENGINE
# ---------------------------------------------------------------------------
def geocode_city(city_name):
    time.sleep(1.0)  # Throttling to guarantee OpenStreetMap SLA safety boundaries
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(city_name)}&limit=1"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return f"Latitude: {data['lat']}, Longitude: {data['lon']}"
        return "City coordinates could not be resolved."
    except Exception as e:
        return f"Geocoding error: {str(e)}"

# ---------------------------------------------------------------------------
# 5. SIDEBAR PARAMETERS INTERFACE
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
gemini_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.title("Trip Details")
destination = st.sidebar.text_input("Where do you want to go?", placeholder="e.g., Tokyo, Mumbai, Paris")
duration = st.sidebar.slider("Trip Duration (Days)", min_value=1, max_value=7, value=3)
travel_style = st.sidebar.selectbox("Travel Style", ["Adventure", "Cultural/Historical", "Relaxing", "Budget-Friendly", "Luxury"])

# ---------------------------------------------------------------------------
# 6. PIPELINE ORCHESTRATION LAYER (RUN EXECUTION)
# ---------------------------------------------------------------------------
if gemini_key:
    client = OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    st.sidebar.success("🔑 System connected to Gemini Gateway Proxy.")
    
    if st.sidebar.button("Generate Itinerary", type="primary"):
        if not destination:
            st.error("Please specify a target destination location in the sidebar configuration.")
        else:
            with st.spinner(f"AI Agent mining structured spatial profiles for {destination}..."):
                # Fetching historical mathematical weights from previous sessions
                active_boosts = calculate_poi_boosts(destination)
                
                # Instructing Gemini to output structured JSON with an explicit POI array
                system_instruction = (
                    "You are a corporate travel agent tracking real-time user engagement metrics. "
                    "You MUST respond ONLY with a raw JSON object containing two top-level keys: "
                    "'text_itinerary' (containing a beautiful, rich markdown travel guide) and "
                    "'pois' (a clean array of objects representing specific locations visited). "
                    "Each POI object must contain exactly: 'id' (a unique lowercase snake_case string), "
                    "'name' (the readable spot name), and 'description' (a brief sentence explaining why it fits)."
                )
                
                # Injecting historical telemetry adjustments directly into user context window
                user_prompt = (
                    f"Build a detailed {duration}-day roadmap for {destination} tailored around a {travel_style} travel profile. "
                    f"Historical user preferences for {destination} indicate these current ranking weights: {json.dumps(active_boosts)}. "
                    "Prioritize recommending and expanding on high-scoring spots, and avoid downvoted items."
                )
                
                try:
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2
                    )
                    
                    # Target content extraction
                    raw_payload = response.choices[0].message.content
                    parsed_json = json.loads(raw_payload)
                    
                    # Store variables directly into session memory state
                    st.session_state.itinerary_data = parsed_json
                    st.session_state.geo_info = geocode_city(destination)
                    
                except Exception as err:
                    st.error(f"Execution Error Intercepted: {str(err)}")
else:
    st.sidebar.warning("⚠️ Setup Required: Please input your Gemini API Key above.")

# ---------------------------------------------------------------------------
# 7. PRODUCTION FRONTEND DESIGN & LAYOUT LAYER
# ---------------------------------------------------------------------------
if st.session_state.itinerary_data:
    # Setup our asymmetric responsive visual columns grid
    left_col, right_col = st.columns([4, 3], gap="large")
    
    with left_col:
        ui_tab_plan, ui_tab_pois, ui_tab_metrics = st.tabs(["📋 Travel Plan", "🗳️ Vote on Places", "📊 Model Insights"])
        
        with ui_tab_plan:
            st.subheader(f"Custom Strategic Roadmap: {destination}")
            with st.container(border=True):
                # Safely prints text itinerary generated from the state file
                st.markdown(st.session_state.itinerary_data.get("text_itinerary", "No textual itinerary extracted."))
                
        with ui_tab_pois:
            st.subheader("Interactive Reinforcement Feedback Panel")
            st.write("Help train our destination selection algorithms over time by upvoting or downvoting these places:")
            
            pois_list = st.session_state.itinerary_data.get("pois", [])
            current_boost_snapshot = calculate_poi_boosts(destination)
            
            for index, poi in enumerate(pois_list):
                poi_id = poi.get("id", f"poi_{index}")
                poi_name = poi.get("name", "Unknown Location")
                poi_desc = poi.get("description", "")
                
                # Display individual POI inside a card container structure
                with st.container(border=True):
                    col_info, col_up, col_down, col_score = st.columns([5, 1, 1, 2])
                    with col_info:
                        st.markdown(f"**{poi_name}** \n*{poi_desc}*")
                    
                    # Unique widget keys prevent interaction collisions
                    with col_up:
                        if st.button("👍", key=f"up_{poi_id}_{index}"):
                            save_feedback(destination, poi_id, "up")
                            st.toast(f"Upvoted {poi_name}! Re-run to update rankings.", icon="🚀")
                            st.rerun()
                            
                    with col_down:
                        if st.button("👎", key=f"down_{poi_id}_{index}"):
                            save_feedback(destination, poi_id, "down")
                            st.toast(f"Downvoted {poi_name}.", icon="📉")
                            st.rerun()
                            
                    with col_score:
                        score = current_boost_snapshot.get(poi_id, 0.0)
                        # Color code metrics for scannability
                        if score > 0:
                            st.markdown(f"<span style='color:green'>Boost: +{score:.2f}</span>", unsafe_allow_html=True)
                        elif score < 0:
                            st.markdown(f"<span style='color:red'>Boost: {score:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:gray'>Boost: 0.00</span>", unsafe_allow_html=True)

        with ui_tab_metrics:
            st.subheader("Historical Analytics File Telemetry (`feedback.jsonl`)")
            if os.path.exists(FEEDBACK_FILE):
                with open(FEEDBACK_FILE, "r") as f:
                    lines = [json.loads(l) for l in f.readlines() if l.strip()]
                st.dataframe(pd.DataFrame(lines), use_container_width=True)
            else:
                st.info("No system logging interactions have been recorded yet.")

    with right_col:
        st.subheader("📍 Real-Time Location Verification")
        geo = st.session_state.geo_info
        
        if geo and "Latitude" in geo:
            try:
                lat_val = float(geo.split("Latitude: ")[1].split(",")[0].strip())
                lon_val = float(geo.split("Longitude: ")[1].strip())
                map_df = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=lat_val, longitude=lon_val, zoom=11, pitch=35),
                    layers=[pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_radius=650, get_color='[225, 29, 72, 190]')]
                ))
                st.caption(f"**Geospatial Tracking Status:** Coordinates resolved cleanly to ({lat_val}, {lon_val}).")
            except Exception as e:
                st.error(f"Spatial Error: {str(e)}")
        else:
            st.info("System is idling. Provide parameters in the configuration panel to draw viewport graphics.")
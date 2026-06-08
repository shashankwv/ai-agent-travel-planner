import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import time
import json
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. SETUP & PATH MANAGEMENT
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise AI Travel Agent", layout="wide", page_icon="🗺️")
st.title("🗺️ Enterprise AI Travel Agent")
st.write("Robust production framework featuring defensive error tracking and exponential backoff loops.")

FEEDBACK_FILE = "data/feedback.jsonl"
os.makedirs("data", exist_ok=True)

USER_AGENT = "trip-planner-capstone/1.0 (mycustomtestproject99@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT}

# Initialize session storage arrays
if "itinerary_data" not in st.session_state:
    st.session_state.itinerary_data = None
if "geo_coords" not in st.session_state:
    st.session_state.geo_coords = None

# ---------------------------------------------------------------------------
# 2. DEFENSIVE FEEDBACK LAYER
# ---------------------------------------------------------------------------
def calculate_poi_boosts(city_key):
    boost_map = {}
    normalized_city = city_key.lower().strip()
    if not os.path.exists(FEEDBACK_FILE):
        return boost_map
    try:
        with open(FEEDBACK_FILE, "r") as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    if event.get("city_key") == normalized_city:
                        poi = event.get("poi_id")
                        vote = event.get("vote")
                        current_boost = boost_map.get(poi, 0.0)
                        boost_map[poi] = current_boost + (0.25 if vote == "up" else -0.35)
    except Exception:
        pass  # Defensive failure isolation: feedback errors must never crash the main app
    return boost_map

def save_feedback(city_key, poi_id, vote):
    try:
        feedback_event = {
            "ts": time.time(),
            "city_key": city_key.lower().strip(),
            "poi_id": poi_id,
            "vote": vote
        }
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(feedback_event) + "\n")
    except Exception as e:
        st.error(f"Failed to record persistent feedback log: {str(e)}")

# ---------------------------------------------------------------------------
# 3. DEFENSIVE GEOGRAPHIC ENGINE WITH EXPONENTIAL BACKOFF
# ---------------------------------------------------------------------------
def geocode_city_robust(city_name):
    """
    Queries OpenStreetMap utilizing exponential backoff logic to gracefully 
    manage rate limits (HTTP 429) and network timeouts.
    """
    sanitized_city = requests.utils.quote(city_name.strip())
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={sanitized_city}&limit=1"
    
    # Enforce base rate compliance rule before executing call
    time.sleep(1.0)
    
    for attempt in range(3):
        try:
            # Enforce 15-second strict timeout limit per request
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            # Catch Rate-Limitation triggers instantly
            if response.status_code == 429:
                wait_time = 2 ** attempt
                st.toast(f"Rate limit hit (429). Backing off for {wait_time}s...", icon="⏳")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return {"status": "NO_RESULTS", "msg": f"OpenStreetMap found zero matches for '{city_name}'."}
                
            return {
                "status": "SUCCESS",
                "lat": float(data[0]['lat']),
                "lon": float(data[0]['lon']),
                "display_name": data[0]['display_name']
            }
            
        except requests.exceptions.Timeout:
            wait_time = 2 ** attempt
            if attempt < 2:
                time.sleep(wait_time)
                continue
            return {"status": "TIMEOUT", "msg": "Geocoding server communication timed out."}
            
        except Exception as e:
            if attempt == 2:
                return {"status": "ERROR", "msg": f"Geocoding gateway failure: {str(e)}"}
                
    return {"status": "ERROR", "msg": "Exhausted all available API connection retry slots."}

# ---------------------------------------------------------------------------
# 4. SIDEBAR PANEL CONTROL INGESTION
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
gemini_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.title("Trip Details")
destination_input = st.sidebar.text_input("Where do you want to go?", placeholder="e.g., Tokyo, Mumbai, Paris")
duration = st.sidebar.slider("Trip Duration (Days)", min_value=1, max_value=7, value=3)
travel_style = st.sidebar.selectbox("Travel Style", ["Adventure", "Cultural/Historical", "Relaxing", "Budget-Friendly", "Luxury"])

# ---------------------------------------------------------------------------
# 5. ORCHESTRATION PIPELINE ENGINE (WITH INPUT VALIDATION & TIMEOUTS)
# ---------------------------------------------------------------------------
if gemini_key:
    client = OpenAI(api_key=gemini_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    if st.sidebar.button("Generate Itinerary", type="primary"):
        # STEP 1: Strict User Input Validation Validation Layer
        if not destination_input.strip():
            st.sidebar.error("❌ Destination cannot be empty!")
        elif len(destination_input.strip()) < 3:
            st.sidebar.error("❌ Please enter a valid destination name (minimum 3 characters).")
        else:
            destination = destination_input.strip()
            
            with st.spinner(f"AI Agent mining structured spatial profiles for {destination}..."):
                active_boosts = calculate_poi_boosts(destination)
                
                system_instruction = (
                    "You are a corporate travel agent tracking real-time user engagement metrics. "
                    "You MUST respond ONLY with a raw JSON object containing two top-level keys:\n"
                    "1. 'text_itinerary': (A beautiful, rich markdown travel guide overview)\n"
                    "2. 'pois': (An array of objects for specific locations visited. Example: [{'id': 'spot_id', 'name': 'Spot Name', 'description': 'Why it fits'}]).\n\n"
                    "Edge Case Behavior: If the location is completely invalid, fictitious, or lacks POI data, "
                    "return an empty array for 'pois' and explain the issue inside the 'text_itinerary' string field."
                )
                
                user_prompt = (
                    f"Build a detailed {duration}-day roadmap for {destination} tailored around a {travel_style} profile. "
                    f"Historical preferences for {destination} reflect these algorithm boost scores: {json.dumps(active_boosts)}. "
                    "Incorporate high-scoring items and do not return downvoted items."
                )
                
                raw_output = ""
                try:
                    # STEP 2: Main Inference Call with Timeout Handling
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2
                    )
                    raw_output = response.choices[0].message.content
                    
                    # STEP 3: Strategic JSON Parsing Validation Block
                    if not raw_output:
                        raise ValueError("Remote gateway returned an empty inference payload.")
                        
                    parsed_json = json.loads(raw_output)
                    
                    # Store to session state memory only after passing validation check
                    st.session_state.itinerary_data = parsed_json
                    
                    # STEP 4: Robust Geocoding Trigger Verification
                    st.session_state.geo_coords = geocode_city_robust(destination)
                    
                except json.JSONDecodeError as json_err:
                    st.error(f"❌ Critical JSON Parsing Error: Malformed API structure returned from server.")
                    st.info("The application caught this failure gracefully to prevent a UI crash. Inspect the raw response data below:")
                    with st.expander("🔍 View Raw Model Response Content", expanded=True):
                        st.code(raw_output if raw_output else "No data received.")
                    st.stop()
                    
                except Exception as api_err:
                    if "503" in str(api_err) or "UNAVAILABLE" in str(api_err):
                        st.error("⚠️ The AI engine is currently overloaded (HTTP 503). Please wait 5 seconds and resubmit your request.")
                    else:
                        st.error(f"❌ Connection Pipeline Error: {str(api_err)}")
                    st.stop()
else:
    st.sidebar.warning("⚠️ Setup Required: Please input your Gemini API Key above.")

# ---------------------------------------------------------------------------
# 6. ASYMMETRIC UI DISPLAY LAYER (HANDLING COMPONENT EDGE CASES)
# ---------------------------------------------------------------------------
if st.session_state.itinerary_data:
    left_col, right_col = st.columns([4, 3], gap="large")
    
    with left_col:
        ui_tab_plan, ui_tab_pois = st.tabs(["📋 Travel Plan", "🗳️ Vote on Places"])
        
        with ui_tab_plan:
            st.subheader(f"Custom Strategic Roadmap: {destination_input}")
            with st.container(border=True):
                st.markdown(st.session_state.itinerary_data.get("text_itinerary", "No textual plan generated."))
                
        with ui_tab_pois:
            st.subheader("Interactive Reinforcement Feedback Panel")
            pois_list = st.session_state.itinerary_data.get("pois", [])
            
            # Handle Edge Case: If city contains zero matching POI objects
            if not pois_list:
                st.info("ℹ️ No structured Points of Interest objects were returned for this location profile.")
            else:
                current_boost_snapshot = calculate_poi_boosts(destination_input)
                for index, poi in enumerate(pois_list):
                    poi_id = poi.get("id", f"poi_{index}")
                    poi_name = poi.get("name", "Unknown Spot")
                    poi_desc = poi.get("description", "No descriptive metrics provided.")
                    
                    with st.container(border=True):
                        col_info, col_up, col_down, col_score = st.columns([5, 1, 1, 2])
                        with col_info:
                            st.markdown(f"**{poi_name}** \n\n*{poi_desc}*")
                        with col_up:
                            if st.button("👍", key=f"up_{poi_id}_{index}"):
                                save_feedback(destination_input, poi_id, "up")
                                st.rerun()
                        with col_down:
                            if st.button("👎", key=f"down_{poi_id}_{index}"):
                                save_feedback(destination_input, poi_id, "down")
                                st.rerun()
                        with col_score:
                            score = current_boost_snapshot.get(poi_id, 0.0)
                            if score > 0:
                                st.markdown(f"<span style='color:green'>Boost: +{score:.2f}</span>", unsafe_allow_html=True)
                            elif score < 0:
                                st.markdown(f"<span style='color:red'>Boost: {score:.2f}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color:gray'>Boost: 0.00</span>", unsafe_allow_html=True)

    with right_col:
        st.subheader("📍 Real-Time Spatial Verification")
        geo_state = st.session_state.geo_coords
        
        if geo_state:
            # Handle Edge Case Scenario A: Geocoding found zero coordinate results
            if geo_state.get("status") == "NO_RESULTS":
                st.warning(f"⚠️ Map Viewport Deactivated: {geo_state.get('msg')}")
                st.info("The text itinerary is still completely accessible, but the city name could not be resolved on global tracking grids.")
                
            # Handle Edge Case Scenario B: Network Lookup completely failed or timed out
            elif geo_state.get("status") in ["TIMEOUT", "ERROR"]:
                st.error(f"❌ Spatial Engine offline: {geo_state.get('msg')}")
                st.caption("Please verify your internet connection or check if the external geocoding server is experiencing downtime.")
                
            # Handle Success Path: Render hardware accelerated map view smoothly
            elif geo_state.get("status") == "SUCCESS":
                try:
                    lat_val = geo_state["lat"]
                    lon_val = geo_state["lon"]
                    map_df = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                    
                    st.pydeck_chart(pdk.Deck(
                        initial_view_state=pdk.ViewState(latitude=lat_val, longitude=lon_val, zoom=11, pitch=35),
                        layers=[pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_radius=650, get_color='[225, 29, 72, 190]')]
                    ))
                    st.caption(f"Confirmed Tracking: **{geo_state['display_name']}**")
                except Exception as map_err:
                    st.error(f"Failed to compile Pydeck mapping layer: {str(map_err)}")
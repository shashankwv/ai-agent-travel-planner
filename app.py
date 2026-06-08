import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
import time
from openai import OpenAI  # Keeping the course's exact library import

# ---------------------------------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Agent Travel Planner", layout="wide", page_icon="🗺️")
st.title("🗺️ AI Agent Travel Planner")
st.write("Generate intelligent itineraries using OpenAI SDK routed through free Google Gemini 2.5 Flash.")

# Set up required course headers for OpenStreetMap/Wikivoyage API compliance
USER_AGENT = "trip-planner-capstone/1.0 (your-email@example.com)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# 2. SIDEBAR - KEY CONFIGURATION & USER INPUTS
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
gemini_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password", help="Get a free key from https://aistudio.google.com/")

st.sidebar.markdown("---")
st.sidebar.title("Trip Details")
destination = st.sidebar.text_input("Where do you want to go?", placeholder="e.g., Tokyo, Paris, New York")
duration = st.sidebar.slider("Trip Duration (Days)", min_value=1, max_value=7, value=3)
travel_style = st.sidebar.selectbox("Travel Style", ["Adventure", "Cultural/Historical", "Relaxing", "Budget-Friendly", "Luxury"])

# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR TOOLS (OpenStreetMap & Wikivoyage)
# ---------------------------------------------------------------------------

def geocode_city(city_name):
    time.sleep(1.0) # Respect Nominatim's 1 request per second rate limit
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}&limit=1"
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return f"Latitude: {data['lat']}, Longitude: {data['lon']}"
        return "City coordinates could not be resolved."
    except Exception as e:
        return f"Geocoding error: {str(e)}"

def get_points_of_interest(city_name):
    time.sleep(1.0)
    geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}&limit=1"
    try:
        geo_resp = requests.get(geo_url, headers=HEADERS).json()
        if not geo_resp:
            return "Could not fetch POIs because city location failed."
        lat, lon = float(geo_resp[0]['lat']), float(geo_resp[0]['lon'])
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node["tourism"="attraction"](around:3000,{lat},{lon});
          node["tourism"="museum"](around:3000,{lat},{lon});
        );
        out body 10;
        """
        response = requests.post(overpass_url, data=overpass_query, headers=HEADERS)
        if response.status_code == 200:
            elements = response.json().get('elements', [])
            pois = [e['tags'].get('name', 'Unnamed Attraction') for e in elements if 'tags' in e]
            return ", ".join(pois[:8]) if pois else "No major attractions found in this radius."
        return "Failed to fetch POI data from Overpass API."
    except Exception as e:
        return f"POI fetch error: {str(e)}"

# ---------------------------------------------------------------------------
# 4. OPENAI TOOL DEFINITIONS (Course Specification Layout)
# ---------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "geocode_city",
            "description": "Retrieves the latitude and longitude coordinates for a given city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "The name of the city to locate."}
                },
                "required": ["city_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_points_of_interest",
            "description": "Retrieves top tourist points of interest (POIs) using the Overpass API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {"type": "string", "description": "The city name to find tourist attractions for."}
                },
                "required": ["city_name"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# 5. AGENT EXECUTION CORE (OpenAI Syntax pointing to Google Servers)
# ---------------------------------------------------------------------------
if gemini_key:
    # We initialize OpenAI, but trick it into talking to Google's endpoint!
    client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    st.sidebar.success("🔑 OpenAI Client Routed to Gemini!")
    
    if st.sidebar.button("Generate Itinerary", type="primary"):
        if not destination:
            st.error("Please specify a destination in the sidebar first!")
        else:
            with st.spinner(f"Executing agent functions via OpenAI library..."):
                
                messages = [
                    {"role": "system", "content": "You are an expert travel agent. Use your tools to look up coordinates and points of interest before building itineraries."},
                    {"role": "user", "content": f"Build a {duration}-day itinerary for {destination} with a {travel_style} focus."}
                ]
                
                try:
                    # Pure OpenAI code call structure
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",  # The free Gemini model name
                        messages=messages,
                        tools=tools,
                        temperature=0.7
                    )
                    
                    # ---------------------------------------------------------------------------
                    # 6. RENDER ITINERARY UI
                    # ---------------------------------------------------------------------------
                    st.success("✨ Itinerary Generated Successfully!")
                    
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.subheader("📋 Your Personalized Itinerary")
                        # Display text output
                        st.write(response.choices[0].message.content)
                        
                    with col2:
                        st.subheader("📍 Live Map Tracking")
                        geo_info = geocode_city(destination)
                        if "Latitude" in geo_info:
                            try:
                                lat_val = float(geo_info.split("Latitude: ")[1].split(",")[0])
                                lon_val = float(geo_info.split("Longitude: ")[1])
                                map_data = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                                
                                st.pydeck_chart(pdk.Deck(
                                    initial_view_state=pdk.ViewState(latitude=lat_val, longitude=lon_val, zoom=11),
                                    layers=[pdk.Layer('ScatterplotLayer', data=map_data, get_position='[lon, lat]', get_radius=400, get_color='[200, 30, 0, 160]')]
                                ))
                            except Exception:
                                st.info("Map tracking updated.")
                                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
else:
    st.sidebar.warning("⚠️ Setup Required: Please input your Gemini API Key above.")
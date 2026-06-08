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

# Set up required course headers for OpenStreetMap API compliance
USER_AGENT = "trip-planner-capstone/1.0 (mycustomtestproject99@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------------------------------------------------------
# 2. SIDEBAR - KEY CONFIGURATION & USER INPUTS
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
gemini_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password", help="Get a free key from https://aistudio.google.com/")

st.sidebar.markdown("---")
st.sidebar.title("Trip Details")
destination = st.sidebar.text_input("Where do you want to go?", placeholder="e.g., Tokyo, Mumbai, Paris")
duration = st.sidebar.slider("Trip Duration (Days)", min_value=1, max_value=7, value=3)
travel_style = st.sidebar.selectbox("Travel Style", ["Adventure", "Cultural/Historical", "Relaxing", "Budget-Friendly", "Luxury"])

# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS FOR TOOLS (OpenStreetMap API)
# ---------------------------------------------------------------------------

def geocode_city(city_name):
    time.sleep(1.0)  # Respect Nominatim's 1 request per second rate limit
    
    # URL encode the city name automatically to handle multi-word inputs cleanly
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(city_name)}&limit=1"
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            # Pass back a clean dictionary string format that our map parsing block can handle
            return f"Latitude: {data['lat']}, Longitude: {data['lon']}"
        return "City coordinates could not be resolved."
    except Exception as e:
        return f"Geocoding error: {str(e)}"

# ---------------------------------------------------------------------------
# 4. AGENT EXECUTION CORE (OpenAI Syntax pointing to Google Servers)
# ---------------------------------------------------------------------------
if gemini_key:
    # Initialize OpenAI client pointing to Google's OpenAI-compatible translation layer
    client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    st.sidebar.success("🔑 OpenAI Client Routed to Gemini!")
    
    if st.sidebar.button("Generate Itinerary", type="primary"):
        if not destination:
            st.error("Please specify a destination in the sidebar first!")
        else:
            with st.spinner(f"Generating your {destination} travel plan..."):
                
                # Optimized prompt to ensure Gemini returns immediate markdown text content
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert travel agent. When asked to plan a trip, write a highly descriptive, beautifully styled markdown itinerary day-by-day immediately. Use bolding, bullet points, and clear headers."
                    },
                    {
                        "role": "user", 
                        "content": f"Please build a detailed {duration}-day itinerary for {destination} with a focus on {travel_style} travel style."
                    }
                ]
                
                try:
                    # Removed tools=[] here to stop Gemini from trying to call empty back-end functions
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",  
                        messages=messages,
                        temperature=0.7
                    )
                    
                    # ---------------------------------------------------------------------------
                    # 5. EXTRACTION LAYER
                    # ---------------------------------------------------------------------------
                    message_obj = response.choices[0].message
                    itinerary_text = None
                    
                    if hasattr(message_obj, 'content') and message_obj.content:
                        itinerary_text = message_obj.content
                    elif isinstance(message_obj, dict) and 'content' in message_obj:
                        itinerary_text = message_obj['content']
                    elif hasattr(message_obj, 'text') and message_obj.text:
                        itinerary_text = message_obj.text
                        
                    # ---------------------------------------------------------------------------
                    # 6. RENDER ITINERARY UI
                    # ---------------------------------------------------------------------------
                    if itinerary_text:
                        st.success("✨ Itinerary Generated Successfully!")
                        
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            st.subheader("📋 Your Personalized Itinerary")
                            st.markdown(itinerary_text)  
                            
                        with col2:
                            st.subheader("📍 Live Map Tracking")
                            geo_info = geocode_city(destination)
                            if "Latitude" in geo_info:
                                try:
                                    lat_val = float(geo_info.split("Latitude: ")[1].split(",")[0])
                                    lon_val = float(geo_info.split("Longitude: ")[1])
                                    map_data = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                                    
                                    st.pydeck_chart(pdk.Deck(
                                        initial_view_state=pdk.ViewState(latitude=lat_val, longitude=lon_val, zoom=11, pitch=30),
                                        layers=[pdk.Layer('ScatterplotLayer', data=map_data, get_position='[lon, lat]', get_radius=500, get_color='[200, 30, 0, 160]')]
                                    ))
                                except Exception:
                                    st.info("Map centered at destination coordinates.")
                            
                            st.markdown("---")
                            st.caption("🤖 **Agent Debug Logs:**")
                            st.info(f"**Coordinates Found:** {geo_info}")
                    else:
                        st.error("The model responded, but no plain text content could be extracted. Please try clicking generate again.")
                                
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        st.error("⚠️ Google's free servers are currently overloaded. Please wait 10 seconds and click 'Generate Itinerary' again!")
                    else:
                        st.error(f"An error occurred: {str(e)}")
else:
    st.sidebar.warning("⚠️ Setup Required: Please input your Gemini API Key above.")
    st.info("👈 Enter your free Gemini API key in the sidebar configuration panel to test the travel agent application.")
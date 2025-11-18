import streamlit as st
import os
from src.stock_agent import StockAgent

st.set_page_config(page_title="Stock Analysis Agent", page_icon="📈", layout="wide")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = StockAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📈 Stock Agent")
    st.markdown("""
    This agent uses **NVIDIA Nemotron** to analyze stock market data.
    
    **Capabilities:**
    - 💰 Get current stock prices
    - 🏢 Get company information
    - 📊 Plot historical price charts
    - 🎯 Simulate "What If?" investment scenarios
    
    **Example Queries:**
    - "What is the price of NVDA?"
    - "Tell me about Microsoft."
    - "Plot the 1-year history of Apple."
    - "What if I invested $500 in NVDA 1 year ago?"
    - "Simulate investing $1000 in Tesla 2 years ago."
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent.clear_history()
        st.rerun()

    # API Key check
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("⚠️ NVIDIA_API_KEY not found!")
        st.info("Please set the environment variable before running.")

# Main Chat Interface
st.title("💬 Stock Analysis Assistant")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check for chart data in the content
        content = message["content"]
        chart_data = None
        
        if "CHART_DATA:" in content:
            # Extract the chart data and clean the content
            parts = content.split("CHART_DATA:")
            content = parts[0].strip()
            if len(parts) > 1:
                chart_data = parts[1].strip()
        
        st.markdown(content)
        
        if chart_data:
            try:
                import plotly.graph_objects as go
                import json
                fig = go.Figure(json.loads(chart_data))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# User Input
if prompt := st.chat_input("Ask about stocks..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, chart_data = st.session_state.agent.chat(prompt)
            
            # Display the text response
            st.markdown(response)
            
            # Display the chart if one was generated
            if chart_data:
                try:
                    import plotly.graph_objects as go
                    import json
                    # Parse the JSON and create figure
                    fig = go.Figure(json.loads(chart_data))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying chart: {e}")
            
            # Store in session (we'll store chart data separately)
            content_to_store = response
            if chart_data:
                content_to_store += f"\n\nCHART_DATA:{chart_data}"
            
            st.session_state.messages.append({"role": "assistant", "content": content_to_store})

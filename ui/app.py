import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Get the API URL from an environment variable, with a fallback for local development
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Set up the Streamlit page
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-result {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
</style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.markdown(
    '<h1 class="main-header">🌷 Iris Species Predictor</h1>', unsafe_allow_html=True
)
st.markdown(
    "**Predict the species of an Iris flower based on its physical characteristics**"
)

# Sidebar for model information
st.sidebar.header("🔧 Model Information")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("status") == "healthy":
            st.sidebar.success("✅ API is healthy")
        else:
            st.sidebar.error("❌ API is unhealthy")
    else:
        st.sidebar.error("❌ API connection failed")

    # Get model info
    info_response = requests.get(f"{API_URL}/model-info", timeout=5)
    if info_response.status_code == 200:
        model_info = info_response.json()
        if model_info.get("model_loaded"):
            st.sidebar.info(
                f"**Model Type:** {model_info.get('model_type', 'Unknown')}"
            )
        else:
            st.sidebar.warning("⚠️ Model not loaded")

except requests.exceptions.RequestException:
    st.sidebar.error("❌ Cannot connect to API")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Input Features")

    # Create input fields with realistic ranges and helpful descriptions
    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.1,
        step=0.1,
        help="Length of the sepal (outer part of the flower)",
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.5,
        step=0.1,
        help="Width of the sepal (outer part of the flower)",
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=1.4,
        step=0.1,
        help="Length of the petal (inner part of the flower)",
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=0.2,
        step=0.1,
        help="Width of the petal (inner part of the flower)",
    )

    # Display current inputs
    st.subheader("📋 Current Input Values")
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.header("🎯 Prediction Results")

    # Prediction button
    if st.button("🔮 Predict Species", type="primary", use_container_width=True):
        try:
            # Make a POST request to the FastAPI endpoint
            with st.spinner("Making prediction..."):
                response = requests.post(
                    f"{API_URL}/predict", json=input_data, timeout=10
                )
                response.raise_for_status()

            # Get the prediction from the response
            prediction = response.json()
            predicted_species = prediction["predicted_species"]
            confidence = prediction["confidence"]
            predicted_class = prediction["predicted_class"]

            # Display the result with confidence
            st.markdown(
                f"""
            <div class="prediction-result success">
                <h3>🌸 Predicted Species: {predicted_species.title()}</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Class:</strong> {predicted_class}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Create confidence visualization
            fig_confidence = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Prediction Confidence (%)"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_confidence.update_layout(height=300)
            st.plotly_chart(fig_confidence, use_container_width=True)

            # Display species images and information
            species_info = {
                "setosa": {
                    "image": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_syberyjski_Iris_sibirica.jpg",  # noqa: E501
                    "description": (
                        "Iris Setosa is characterized by shorter petals "
                        "and is commonly found in Arctic regions."
                    ),
                },
                "versicolor": {
                    "image": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",  # noqa: E501
                    "description": (
                        "Iris Versicolor has medium-sized petals "
                        "and is native to eastern North America."
                    ),
                },
                "virginica": {
                    "image": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",  # noqa: E501
                    "description": (
                        "Iris Virginica has the largest petals "
                        "and is found in eastern North America."
                    ),
                },
            }

            if predicted_species in species_info:
                st.image(
                    species_info[predicted_species]["image"],
                    caption=f"Iris {predicted_species.title()}",
                    width=300,
                )
                st.info(species_info[predicted_species]["description"])

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to the API: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

# Feature visualization
st.header("📈 Feature Visualization")

# Create a radar chart for the input features
categories = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
values = [sepal_length, sepal_width, petal_length, petal_width]

fig_radar = go.Figure()

fig_radar.add_trace(
    go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Current Input",
        line_color="blue",
    )
)

fig_radar.update_layout(
    polar={"radialaxis": {"visible": True, "range": [0, 8]}},
    showlegend=True,
    title="Feature Profile",
)

st.plotly_chart(fig_radar, use_container_width=True)

# Information section
st.header("ℹ️ About the Iris Dataset")
st.markdown("""
The Iris dataset is a classic dataset in machine learning, introduced by
Ronald Fisher in 1936. It contains measurements of four features for three
species of Iris flowers:

- **Sepal Length & Width**: The outer parts of the flower
- **Petal Length & Width**: The inner parts of the flower
- **Species**: Setosa, Versicolor, or Virginica

The model uses these four measurements to predict which species of Iris
flower you're looking at.
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI")

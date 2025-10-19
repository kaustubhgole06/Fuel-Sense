import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="FuelSense: Vehicle Fuel Efficiency Optimizer",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .suggestion-box {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffa500;
        margin: 1rem 0;
    }
    .tips-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #32cd32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_model():
    """Load dataset and trained model"""
    try:
        # Try to load the saved model
        model = joblib.load('models/best_mpg_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        return model, scaler, True
    except:
        # If model files don't exist, return None
        return None, None, False

def create_sample_data():
    """Create sample data for demonstration if real data isn't available"""
    np.random.seed(42)
    n_samples = 398
    
    data = {
        'mpg': np.random.normal(23, 8, n_samples),
        'cylinders': np.random.choice([3, 4, 5, 6, 8], n_samples),
        'displacement': np.random.normal(200, 100, n_samples),
        'horsepower': np.random.normal(100, 40, n_samples),
        'weight': np.random.normal(2900, 800, n_samples),
        'acceleration': np.random.normal(15, 3, n_samples),
        'model_year': np.random.randint(70, 83, n_samples),
        'origin': np.random.choice([1, 2, 3], n_samples)
    }
    
    return pd.DataFrame(data)

def predict_mpg_simple(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    """Simple MPG prediction function using statistical relationships"""
    # Base MPG calculation using empirical relationships
    base_mpg = 50  # Starting point
    
    # Adjust for weight (strongest factor)
    weight_factor = -0.005 * (weight - 2500)
    
    # Adjust for horsepower
    hp_factor = -0.05 * (horsepower - 100)
    
    # Adjust for displacement  
    disp_factor = -0.02 * (displacement - 150)
    
    # Adjust for cylinders
    cyl_factor = -2 * (cylinders - 4)
    
    # Adjust for model year (newer is better)
    year_factor = 0.3 * (model_year - 75)
    
    # Adjust for origin (Japanese cars typically more efficient)
    origin_factor = 2 if origin == 3 else (1 if origin == 2 else 0)
    
    predicted_mpg = base_mpg + weight_factor + hp_factor + disp_factor + cyl_factor + year_factor + origin_factor
    
    return max(10, min(50, predicted_mpg))  # Clamp between reasonable bounds

def suggest_improvements_simple(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    """Generate improvement suggestions"""
    current_mpg = predict_mpg_simple(cylinders, displacement, horsepower, weight, acceleration, model_year, origin)
    suggestions = []
    
    # Weight reduction
    new_weight = weight * 0.9
    weight_improved_mpg = predict_mpg_simple(cylinders, displacement, horsepower, new_weight, acceleration, model_year, origin)
    weight_improvement = weight_improved_mpg - current_mpg
    if weight_improvement > 0.5:
        suggestions.append(f"🔹 Reducing weight by 10% could improve MPG by {weight_improvement:.1f}")
    
    # Horsepower optimization
    new_horsepower = horsepower * 0.9
    hp_improved_mpg = predict_mpg_simple(cylinders, displacement, new_horsepower, weight, acceleration, model_year, origin)
    hp_improvement = hp_improved_mpg - current_mpg
    if hp_improvement > 0.3:
        suggestions.append(f"🔹 Reducing horsepower by 10% could improve MPG by {hp_improvement:.1f}")
    
    # Displacement optimization
    new_displacement = displacement * 0.9
    disp_improved_mpg = predict_mpg_simple(cylinders, new_displacement, horsepower, weight, acceleration, model_year, origin)
    disp_improvement = disp_improved_mpg - current_mpg
    if disp_improvement > 0.3:
        suggestions.append(f"🔹 Reducing displacement by 10% could improve MPG by {disp_improvement:.1f}")
    
    return suggestions

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">⛽ FuelSense: Vehicle Fuel Efficiency Optimizer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        Predict your vehicle's fuel efficiency and discover optimization strategies using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, scaler, model_loaded = load_data_and_model()
    
    # Sidebar for inputs
    st.sidebar.header("🚗 Vehicle Specifications")
    st.sidebar.markdown("Adjust the parameters below to predict fuel efficiency:")
    
    # Input sliders
    cylinders = st.sidebar.slider("Number of Cylinders", 3, 8, 4, help="Number of cylinders in the engine")
    displacement = st.sidebar.slider("Engine Displacement (cu in)", 70, 450, 200, help="Engine displacement in cubic inches")
    horsepower = st.sidebar.slider("Horsepower", 50, 250, 100, help="Engine horsepower")
    weight = st.sidebar.slider("Weight (lbs)", 1500, 5000, 2800, help="Vehicle weight in pounds")
    acceleration = st.sidebar.slider("Acceleration (0-60 mph)", 8.0, 25.0, 15.0, 0.1, help="Time to accelerate from 0 to 60 mph")
    model_year = st.sidebar.slider("Model Year", 70, 82, 76, help="Model year (70 = 1970, 82 = 1982)")
    
    origin_map = {1: "USA", 2: "Europe", 3: "Japan"}
    origin_name = st.sidebar.selectbox("Origin", list(origin_map.values()))
    origin = [k for k, v in origin_map.items() if v == origin_name][0]
    
    # Prediction section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Fuel Efficiency Prediction")
        
        # Make prediction
        predicted_mpg = predict_mpg_simple(cylinders, displacement, horsepower, weight, acceleration, model_year, origin)
        
        # Display prediction
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Predicted Miles Per Gallon (MPG)",
            value=f"{predicted_mpg:.1f}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Efficiency category
        if predicted_mpg >= 30:
            efficiency_category = "🟢 Excellent"
            category_color = "green"
        elif predicted_mpg >= 25:
            efficiency_category = "🟡 Good"
            category_color = "orange"
        elif predicted_mpg >= 20:
            efficiency_category = "🟠 Average"
            category_color = "orange"
        else:
            efficiency_category = "🔴 Poor"
            category_color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <h3 style="color: {category_color};">Efficiency Rating: {efficiency_category}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_mpg,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "MPG Gauge"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 35
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.header("💡 Optimization Tips")
        
        # Top fuel efficiency tips
        st.markdown("""
        <div class="tips-box">
            <h4>🏆 Top Fuel Efficiency Tips:</h4>
            <ul>
                <li><strong>Reduce Weight:</strong> Every 100 lbs removed can improve MPG by 1-2%</li>
                <li><strong>Engine Efficiency:</strong> Smaller, turbocharged engines often perform better</li>
                <li><strong>Aerodynamics:</strong> Improved design reduces drag significantly</li>
                <li><strong>Maintenance:</strong> Regular tune-ups and proper tire pressure</li>
                <li><strong>Driving Habits:</strong> Smooth acceleration and maintaining steady speeds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Improvement suggestions
    st.header("🔧 Personalized Improvement Suggestions")
    
    suggestions = suggest_improvements_simple(cylinders, displacement, horsepower, weight, acceleration, model_year, origin)
    
    if suggestions:
        st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
        st.markdown("**Based on your vehicle specifications, here are potential improvements:**")
        for suggestion in suggestions:
            st.markdown(suggestion)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Your vehicle is already quite efficient! Consider regular maintenance to maintain optimal performance.")
    
    # Comparison chart
    st.header("📈 Fuel Efficiency Comparison")
    
    # Create comparison data
    comparison_data = []
    origins = ["USA", "Europe", "Japan"]
    cylinders_range = [4, 6, 8]
    
    for orig in origins:
        for cyl in cylinders_range:
            orig_code = [k for k, v in origin_map.items() if v == orig][0]
            mpg = predict_mpg_simple(cyl, displacement, horsepower, weight, acceleration, model_year, orig_code)
            comparison_data.append({
                'Origin': orig,
                'Cylinders': f"{cyl} Cyl",
                'MPG': mpg
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig_comparison = px.bar(
        comparison_df, 
        x='Origin', 
        y='MPG', 
        color='Cylinders',
        title="MPG Comparison by Origin and Cylinders (Using Your Other Specs)",
        barmode='group'
    )
    fig_comparison.update_layout(height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature impact analysis
    st.header("🎯 Feature Impact Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Weight impact
        weight_range = np.linspace(2000, 4000, 20)
        weight_mpg = [predict_mpg_simple(cylinders, displacement, horsepower, w, acceleration, model_year, origin) for w in weight_range]
        
        fig_weight = px.line(
            x=weight_range, 
            y=weight_mpg,
            title="MPG vs Weight",
            labels={'x': 'Weight (lbs)', 'y': 'MPG'}
        )
        fig_weight.add_vline(x=weight, line_dash="dash", line_color="red", annotation_text="Your Car")
        st.plotly_chart(fig_weight, use_container_width=True)
    
    with col4:
        # Horsepower impact
        hp_range = np.linspace(70, 200, 20)
        hp_mpg = [predict_mpg_simple(cylinders, displacement, hp, weight, acceleration, model_year, origin) for hp in hp_range]
        
        fig_hp = px.line(
            x=hp_range, 
            y=hp_mpg,
            title="MPG vs Horsepower",
            labels={'x': 'Horsepower', 'y': 'MPG'}
        )
        fig_hp.add_vline(x=horsepower, line_dash="dash", line_color="red", annotation_text="Your Car")
        st.plotly_chart(fig_hp, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>FuelSense Dashboard</strong> | Built with Streamlit and Machine Learning</p>
        <p>💡 <em>Optimize your vehicle's fuel efficiency with data-driven insights</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# cyclone_predictor_enhanced.py - Advanced Cyclone Prediction with 4 ML Models

import streamlit as st
import pandas as pd
import folium
import numpy as np
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor

# Helper function to ensure JSON serializable data
def ensure_serializable(data):
    if isinstance(data, (np.integer)):
        return int(data)
    elif isinstance(data, (np.floating)):
        return float(data)
    elif isinstance(data, (np.ndarray)):
        return data.tolist()
    else:
        return data

# Page Configuration
st.set_page_config(
    page_title="Advanced Cyclone Predictor", 
    layout="wide",
    page_icon="🌪️"
)
st.title("🌪️ Advanced Cyclone Size Predictor")

# Load Data with enhanced error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('tropical_cyclone_size.csv', index_col=0)
        
        # Data validation and cleaning
        if df.empty:
            st.error("Dataset is empty. Please check your data file.")
            return None
            
        # Convert coordinates to numeric and handle missing values
        numeric_cols = ['Latitude', 'Longitude', 'Pressure', 'Wind Speed', 'SiR34']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with missing essential values
        df = df.dropna(subset=['Latitude', 'Longitude', 'SiR34'])
        
        if df.empty:
            st.error("No valid data after cleaning. Check your dataset.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# Train Models with enhanced validation
@st.cache_resource
def train_models(_df):
    try:
        # Verify we have all required columns
        required_cols = ['Latitude', 'Longitude', 'Pressure', 'Wind Speed', 'SiR34']
        if not all(col in _df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in _df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
            
        X = _df[['Latitude','Longitude', 'Pressure', 'Wind Speed']]
        y = _df['SiR34']
        
        # Verify we have enough data
        if len(X) < 20:
            st.error(f"Insufficient data points ({len(X)}). Need at least 20.")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=150, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        
        # Neural Network with validation
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Add early stopping to prevent overfitting
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            X_train, y_train, 
            validation_split=0.2,
            epochs=50, 
            batch_size=32, 
            verbose=0,
            callbacks=[early_stop]
        )
        
        nn_pred = model.predict(X_test, verbose=0).flatten()
        nn_mae = mean_absolute_error(y_test, nn_pred)
        nn_r2 = r2_score(y_test, nn_pred)
        
        return {
            'rf': rf,
            'lr': lr,
            'xgb': xgb,
            'nn': model,
            'metrics': {
                'Random Forest': {'MAE': rf_mae, 'R2': rf_r2},
                'Linear Regression': {'MAE': lr_mae, 'R2': lr_r2},
                'XGBoost': {'MAE': xgb_mae, 'R2': xgb_r2},
                'Neural Network': {'MAE': nn_mae, 'R2': nn_r2}
            },
            'features': X.columns.tolist(),
            'history': history  # For plotting training curves
        }
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

models = train_models(df)
if models is None:
    st.stop()

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    
    # Model Selection
    model_type = st.selectbox(
        "Prediction Model",
        ["Random Forest", "Linear Regression", "XGBoost", "Neural Network"],
        index=0,
        help="Select the machine learning model for prediction"
    )
    
    # Cyclone Selection with validation
    available_cyclones = df.index.unique()
    if len(available_cyclones) == 0:
        st.error("No cyclones available in dataset")
        st.stop()
    
    selected_cyclone = st.selectbox(
        "Select Cyclone", 
        available_cyclones,
        help="Choose a cyclone from the dataset"
    )
    
    # Feature Customization
    st.markdown("---")
    st.subheader("Feature Customization")
    use_default_features = st.checkbox("Use Default Features", True)
    
    if not use_default_features:
        selected_features = st.multiselect(
            "Select Features",
            models['features'],
            default=models['features']
        )
    else:
        selected_features = models['features']
    
    # Display Options
    st.markdown("---")
    st.subheader("Map Settings")
    show_actual = st.checkbox("Show Actual Size", True)
    show_predicted = st.checkbox("Show Predicted Size", True)
    zoom_level = st.slider("Zoom Level", 1, 10, 5)
    opacity = st.slider("Circle Opacity", 0.1, 1.0, 0.5)
    
    # Model Evaluation Section
    st.markdown("---")
    st.subheader("Model Performance")
    st.write(f"Dataset size: {len(df)} cyclones")
    
    metrics_df = pd.DataFrame(models['metrics']).T
    st.dataframe(metrics_df.style.format("{:.2f}").highlight_min(
        subset=['MAE'], color='lightgreen').highlight_max(
        subset=['R2'], color='lightgreen'), 
        use_container_width=True)

# Get Selected Cyclone Data with robust extraction
try:
    cyclone_data = df.loc[selected_cyclone]
    if isinstance(cyclone_data, pd.DataFrame):
        cyclone_data = cyclone_data.iloc[0]  # Take first row if multiple matches
    
    actual_size = ensure_serializable(cyclone_data['SiR34'])
    
    # Safely extract and validate coordinates
    lat = ensure_serializable(cyclone_data['Latitude'])
    lon = ensure_serializable(cyclone_data['Longitude'])
    
    # Validate ranges
    if not (-90 <= lat <= 90):
        st.warning(f"Latitude {lat} out of bounds, clamping to valid range")
        lat = max(-90, min(90, lat))
    if not (-180 <= lon <= 180):
        st.warning(f"Longitude {lon} out of bounds, clamping to valid range")
        lon = max(-180, min(180, lon))

except Exception as e:
    st.error(f"Error processing cyclone data: {str(e)}")
    st.stop()

# Prepare input data for prediction with validation
input_data = []
for feature in models['features']:
    try:
        value = ensure_serializable(cyclone_data[feature]) if feature in selected_features else 0.0
        input_data.append(value)
    except Exception as e:
        st.error(f"Invalid value for feature {feature}: {cyclone_data.get(feature, 'N/A')}")
        st.stop()

# Make Predictions with All Models with error handling
predictions = {}
for name, model in [('Random Forest', models['rf']), 
                   ('Linear Regression', models['lr']),
                   ('XGBoost', models['xgb']),
                   ('Neural Network', models['nn'])]:
    try:
        if name == 'Neural Network':
            # Convert input to numpy array with correct shape and type
            nn_input = np.array([input_data], dtype=np.float32)
            pred = float(model.predict(nn_input, verbose=0)[0][0])
        else:
            pred = float(model.predict([input_data])[0])
        predictions[name] = max(0, pred)  # Ensure non-negative prediction
    except Exception as e:
        st.error(f"Error making {name} prediction: {str(e)}")
        predictions[name] = np.nan

# Get current model prediction
predicted_size = predictions.get(model_type, 0)
current_metrics = models['metrics'].get(model_type, {'MAE': 0, 'R2': 0})

# Color mapping for models
model_colors = {
    "Random Forest": "#2ca02c",
    "Linear Regression": "#ff7f0e",
    "XGBoost": "#9467bd",
    "Neural Network": "#d62728"
}

# Create Map with fallback and JSON serialization checks
try:
    # Create base map
    m = folium.Map(
        location=[float(lat), float(lon)],
        zoom_start=int(zoom_level),
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add Actual Circle (if enabled)
    if show_actual:
        folium.Circle(
            location=[float(lat), float(lon)],
            radius=float(actual_size) * 1000,
            color='#3182bd',
            fill=True,
            fill_color='#3182bd',
            fill_opacity=float(opacity),
            popup=f"Actual Size: {float(actual_size):.1f} km",
            tooltip="Actual Size"
        ).add_to(m)

    # Add Predicted Circle (if enabled)
    if show_predicted and not np.isnan(predicted_size):
        folium.Circle(
            location=[float(lat), float(lon)],
            radius=float(predicted_size) * 1000,
            color=model_colors.get(model_type, "#000000"),
            fill=True,
            fill_color=model_colors.get(model_type, "#000000"),
            fill_opacity=float(opacity),
            popup=f"Predicted Size: {float(predicted_size):.1f} km",
            tooltip=f"Predicted ({model_type})"
        ).add_to(m)

    # Add Center Marker with simplified popup
    folium.Marker(
        [float(lat), float(lon)],
        popup=f"Cyclone {str(selected_cyclone)}",
        tooltip="Cyclone Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Display the map in the main column
    with st.container():
        st_folium(m, width=700, height=500)

except Exception as e:
    st.error(f"Failed to create or display map: {str(e)}")
    st.error("Please check your input data for non-numeric values")

# Main Display
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Cyclone Details")
    
    # Metrics cards
    cols = st.columns(2)
    with cols[0]:
        st.metric("Actual Size", f"{actual_size:.1f} km")
    with cols[1]:
        delta = predicted_size - actual_size
        st.metric(
            "Predicted Size", 
            f"{predicted_size:.1f} km",
            delta=f"{delta:.1f} km",
            delta_color="inverse" if delta < 0 else "normal"
        )
    
    # Model performance indicator
    if actual_size > 0:
        accuracy = max(0, 100 - (abs(delta)/actual_size)*100)
        st.progress(int(accuracy), text=f"Prediction Accuracy: {accuracy:.1f}%")
    else:
        st.warning("Cannot calculate accuracy with zero actual size")
    
    # All models prediction comparison
    st.markdown("**All Models Prediction:**")
    comparison_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Size'])
    comparison_df['Actual Size'] = actual_size
    comparison_df['Difference'] = comparison_df['Predicted Size'] - comparison_df['Actual Size']
    st.dataframe(comparison_df.style.format("{:.1f}").apply(
        lambda x: ['background-color: lightgreen' if abs(v) == min(abs(x)) else '' for v in x],
        subset=['Difference']
    ), use_container_width=True)
    
    # Details expander
    with st.expander("Detailed Information"):
        st.write(f"**Cyclone ID:** {selected_cyclone}")
        st.write(f"**Model Used:** {model_type}")
        st.write(f"**Model MAE:** {current_metrics.get('MAE', 'N/A'):.1f} km")
        st.write(f"**Model R²:** {current_metrics.get('R2', 'N/A'):.2f}")
        st.write(f"**Coordinates:** {lat:.4f}°N, {lon:.4f}°E")
        st.write(f"**Wind Speed:** {cyclone_data.get('Wind Speed', 'N/A')} knots")
        st.write(f"**Pressure:** {cyclone_data.get('Pressure', 'N/A')} hPa")
        st.write("**Features Used:**")
        for feature in selected_features:
            st.write(f"- {feature}: {cyclone_data.get(feature, 'N/A')}")

# Advanced Visualizations
st.markdown("---")
st.subheader("Advanced Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Size Distribution", 
    "Wind vs Pressure", 
    "Model Comparison",
    "Feature Analysis",
    "Prediction Errors"
])

with tab1:
    fig = px.histogram(df, x='SiR34', nbins=30, 
                      title='Distribution of Cyclone Sizes',
                      labels={'SiR34': 'Size (km)'},
                      color_discrete_sequence=['#3182bd'])
    fig.add_vline(x=actual_size, line_dash="dash", line_color="red",
                 annotation_text=f"Selected: {actual_size:.1f} km")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(df, x='Wind Speed', y='Pressure', color='SiR34',
                    title='Wind Speed vs Pressure (Colored by Size)',
                    labels={'Wind Speed': 'Wind Speed (knots)', 
                           'Pressure': 'Pressure (hPa)',
                           'SiR34': 'Size (km)'},
                    hover_data=['Latitude', 'Longitude'])
    fig.add_trace(go.Scatter(
        x=[cyclone_data['Wind Speed']],
        y=[cyclone_data['Pressure']],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Selected Cyclone'
    ))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Performance comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(models['metrics'].keys()),
        y=[m['MAE'] for m in models['metrics'].values()],
        name='MAE',
        marker_color=['#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    ))
    fig.add_trace(go.Scatter(
        x=list(models['metrics'].keys()),
        y=[m['R2']*100 for m in models['metrics'].values()],
        name='R² (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='black')
    ))
    fig.update_layout(
        title='Model Performance Comparison',
        yaxis=dict(title='MAE (km)'),
        yaxis2=dict(title='R² (%)', overlaying='y', side='right'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Feature importance/coefficients
    if model_type == "Random Forest":
        importances = models['rf'].feature_importances_
        features = models['features']
        fig = px.bar(x=features, y=importances,
                    title='Random Forest Feature Importance',
                    labels={'x': 'Features', 'y': 'Importance Score'},
                    color=features,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    elif model_type == "Linear Regression":
        coefficients = models['lr'].coef_
        features = models['features']
        fig = px.bar(x=features, y=coefficients,
                    title='Linear Regression Coefficients',
                    labels={'x': 'Features', 'y': 'Coefficient Value'},
                    color=features,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    elif model_type == "XGBoost":
        importances = models['xgb'].feature_importances_
        features = models['features']
        fig = px.bar(x=features, y=importances,
                    title='XGBoost Feature Importance',
                    labels={'x': 'Features', 'y': 'Importance Score'},
                    color=features,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance visualization is not available for Neural Network in this version.")

with tab5:
    # Generate predictions for all data points
    test_data = df.sample(min(100, len(df)), random_state=42)  # Sample for performance
    actuals = test_data['SiR34']
    
    preds = {
        'Random Forest': models['rf'].predict(test_data[models['features']]),
        'Linear Regression': models['lr'].predict(test_data[models['features']]),
        'XGBoost': models['xgb'].predict(test_data[models['features']]),
        'Neural Network': models['nn'].predict(test_data[models['features']].values.astype(np.float32), verbose=0).flatten()
    }
    
    fig = go.Figure()
    for model_name, pred in preds.items():
        fig.add_trace(go.Scatter(
            x=actuals,
            y=pred,
            mode='markers',
            name=model_name,
            marker=dict(color=model_colors[model_name])
        ))
    
    fig.add_trace(go.Scatter(
        x=[min(actuals), max(actuals)],
        y=[min(actuals), max(actuals)],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values Across Models',
        xaxis_title='Actual Size (km)',
        yaxis_title='Predicted Size (km)',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
    **Advanced Cyclone Size Predictor** | Four Machine Learning Models | 
    Made with Streamlit, Plotly, and Scikit-learn
""")

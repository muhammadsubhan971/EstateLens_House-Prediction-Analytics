import streamlit as st
import pickle
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(
    page_title="House Prediction & Analytics",
    page_icon="🏠",
    layout="wide"
)

# Conversion constants
SQFT_PER_MARLA = 272.25
MARLA_PER_KANAL = 20

def sqft_to_marla_kanal(sqft):
    """Convert sqft to Marla/Kanal display format"""
    marla = sqft / SQFT_PER_MARLA
    
    if marla >= MARLA_PER_KANAL:
        kanal = marla / MARLA_PER_KANAL
        if abs(kanal - round(kanal)) < 0.01:
            return f"{int(round(kanal))} Kanal"
        else:
            return f"{kanal:.1f} Kanal"
    else:
        if abs(marla - round(marla)) < 0.01:
            return f"{int(round(marla))} Marla"
        else:
            return f"{marla:.1f} Marla"

def create_size_mapping(df):
    """Create a mapping of display labels to sqft values"""
    unique_sizes = sorted(df['size_sqft'].unique())
    size_mapping = {}
    
    for sqft in unique_sizes:
        display_label = sqft_to_marla_kanal(sqft)
        size_mapping[display_label] = sqft
    
    return size_mapping

def load_map_data():
    """Load and prepare map data"""
    try:
        csv_df = pd.read_csv("merged_output_containment.csv")
        csv_df['latitude'] = pd.to_numeric(csv_df['latitude'], errors='coerce')
        csv_df['longitude'] = pd.to_numeric(csv_df['longitude'], errors='coerce')
        csv_df = csv_df.dropna(subset=['latitude', 'longitude']).copy()
        return csv_df
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        return None

def create_map(df, zoom_level=11):
    """Create folium map with clustered markers"""
    center = [df['latitude'].mean(), df['longitude'].mean()]
    
    # Create map with controls
    m = folium.Map(
        location=center, 
        zoom_start=zoom_level, 
        tiles="OpenStreetMap",
        zoom_control=True,
        scrollWheelZoom=True,
        dragging=True
    )
    
    # Add fullscreen button
    folium.plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Create marker cluster
    marker_cluster = MarkerCluster(
        name='Property Clusters',
        overlay=True,
        control=True,
        icon_create_function=None
    ).add_to(m)
    
    # Add markers
    for _, row in df.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">🏘️ {row.get('original_society', 'N/A')}</h4>
            <hr style="margin: 8px 0;">
            <p style="margin: 5px 0;"><b>💰 Price:</b> {row.get('price', 'N/A')}</p>
            <p style="margin: 5px 0;"><b>📏 Size:</b> {row.get('size_sqft', 'N/A')} sqft</p>
            <p style="margin: 5px 0;"><b>🛏️ Bedrooms:</b> {row.get('bedrooms', 'N/A')}</p>
            <p style="margin: 5px 0;"><b>🚿 Bathrooms:</b> {row.get('bathrooms', 'N/A')}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color='#3498db',
            fill=True,
            fillColor='#3498db',
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row.get('original_society', 'Property')
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def recommend_properties(property_name, df, cosine_sim, top_n=5):
    """
    Recommend similar properties based on cosine similarity
    
    Parameters:
    - property_name: Name of the society to find similar properties for
    - df: DataFrame containing property data
    - cosine_sim: Cosine similarity matrix
    - top_n: Number of recommendations to return
    
    Returns:
    - DataFrame with recommended properties and their details
    """
    try:
        # Get the index of the property that matches the name
        idx = df.index[df['society_extracted'] == property_name].tolist()
        
        if not idx:
            return None
        
        idx = idx[0]

        # Get the pairwise similarity scores with that property
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the properties based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the most similar properties (excluding itself)
        sim_scores = sim_scores[1:top_n+1]

        # Get the property indices
        property_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Create recommendations DataFrame with more details
        recommendations_df = df.iloc[property_indices].copy()
        recommendations_df['similarity_score'] = similarity_scores
        
        # Select relevant columns
        columns_to_show = ['society_extracted', 'bedrooms', 'bathrooms', 'size_sqft', 
                          'feature_category', 'similarity_score']
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in columns_to_show if col in recommendations_df.columns]
        recommendations_df = recommendations_df[available_columns]

        return recommendations_df
    
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return None

# Load data
try:
    with open('df.pkl', "rb") as f:
        df = pickle.load(f)
        
    with open('pipeline.pkl', "rb") as f:
        pipe = pickle.load(f)
    
    # Load recommendation model files
    try:
        # Load cosine similarity matrix (REQUIRED)
        with open('cosine_sim.pkl', "rb") as f:
            cosine_sim = pickle.load(f)
        
        # Load model info (OPTIONAL - for displaying model details)
        try:
            with open('model_info.pkl', "rb") as f:
                model_info = pickle.load(f)
        except FileNotFoundError:
            model_info = None
        
        # Load feature scaler (OPTIONAL - for future predictions)
        try:
            with open('feature_scaler.pkl', "rb") as f:
                feature_scaler = pickle.load(f)
        except FileNotFoundError:
            feature_scaler = None
        
        recommendation_available = True
        
        # Display success message in sidebar
        with st.sidebar:
            st.success("✅ Recommendation System Loaded")
            if model_info:
                st.caption(f"Model Type: {model_info.get('model_type', 'N/A').title()}")
                
    except FileNotFoundError:
        recommendation_available = False
        cosine_sim = None
        model_info = None
        feature_scaler = None
        
        with st.sidebar:
            st.warning("⚠️ Recommendation System Not Available")

except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Property Recommendations", "Analytics Dashboard"])

# Display model info in sidebar if available
if recommendation_available and model_info:
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 Model Info")
        if 'numerical_features' in model_info:
            st.caption(f"📊 Numerical: {len(model_info['numerical_features'])}")
        if 'categorical_features' in model_info:
            st.caption(f"🏷️ Categorical: {len(model_info['categorical_features'])}")
        if 'feature_matrix_shape' in model_info:
            st.caption(f"📈 Matrix: {model_info['feature_matrix_shape']}")

# ========== PRICE PREDICTION PAGE ==========
if page == "Price Prediction":
    st.title("🏠 House Price Prediction")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Property Details")
        
        # Create size mapping
        size_mapping = create_size_mapping(df)
        size_options = list(size_mapping.keys())
        
        # Input fields in columns
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            size_display = st.selectbox('📏 Size', size_options)
            size_sqft = size_mapping[size_display]
            
            bedrooms = st.selectbox('🛏️ Bedrooms', sorted(df['bedrooms'].unique().tolist()))
            
            society_extracted = st.selectbox('🏘️ Society', sorted(df['society_extracted'].unique().tolist()))
        
        with input_col2:
            bathrooms = st.selectbox('🚿 Bathrooms', sorted(df['bathrooms'].unique().tolist()))
            
            feature_category = st.selectbox('⭐ Feature Category', sorted(df['feature_category'].unique().tolist()))
        
        st.markdown("---")
        predict_btn = st.button('🔮 Predict Price', type="primary", use_container_width=True)
        
        if predict_btn:
            test_data = {
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'size_sqft': [size_sqft],
                'society_extracted': [society_extracted],
                'feature_category': [feature_category]
            }
            
            test_df = pd.DataFrame(test_data)
            
            with st.spinner('Calculating price...'):
                prediction = np.expm1(pipe.predict(test_df))
                
                base_price = prediction[0] - 0.50
                org_price = prediction[0] - 0.15
                mid_price = (base_price + org_price) / 2
            
            st.success("✅ Prediction Complete!")
            
            # Display results in cards
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("💵 Minimum Price", f"PKR {base_price:.2f} Cr")
            
            with metric_col2:
                st.metric("💰 Average Price", f"PKR {mid_price:.2f} Cr")
            
            with metric_col3:
                st.metric("💎 Maximum Price", f"PKR {org_price:.2f} Cr")
            
            st.info(f"📊 Estimated price range: **PKR {base_price:.2f} Cr** to **PKR {org_price:.2f} Cr**")
            
            # Add recommendation button
            if recommendation_available:
                st.markdown("---")
                if st.button("🔍 Find Similar Properties", use_container_width=True):
                    st.session_state.recommendation_society = society_extracted
                    st.session_state.show_recommendations = True
                    st.rerun()
    
    with col2:
        st.subheader("📋 Summary")
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h4>How it works:</h4>
            <ol>
                <li>Select property details</li>
                <li>Click predict button</li>
                <li>Get price estimation</li>
                <li>Find similar properties</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model type if available
        if recommendation_available and model_info:
            st.markdown("---")
            st.info(f"🤖 Using {model_info.get('model_type', 'feature-based').replace('_', ' ').title()} Recommendations")

# ========== PROPERTY RECOMMENDATIONS PAGE ==========
elif page == "Property Recommendations":
    st.title("🔍 Property Recommendations")
    st.markdown("---")
    
    if not recommendation_available:
        st.error("❌ Recommendation system is not available. Please ensure 'cosine_sim.pkl' exists.")
        
        st.markdown("### 📥 Setup Instructions")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            **Option 1: Feature-Based (Recommended)**
            
            Run this script to generate recommendations based on property features:
            
            ```bash
            python save_feature_based_recommendation.py
            ```
            
            **Files Generated:**
            - `cosine_sim.pkl` (Required)
            - `model_info.pkl` (Optional)
            - `feature_scaler.pkl` (Optional)
            """)
        
        with col_info2:
            st.markdown("""
            **Option 2: Text-Based**
            
            If you have text descriptions, run:
            
            ```bash
            python save_recommendation_model.py
            ```
            
            **Files Generated:**
            - `cosine_sim.pkl` (Required)
            - `tfidf_vectorizer.pkl` (Optional)
            - `tfidf_matrix.pkl` (Optional)
            """)
        
        st.info("💡 After generating the files, place them in the same directory as this app and refresh the page.")
        
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏘️ Find Similar Properties")
            
            # Check if coming from prediction page
            default_society = None
            if 'recommendation_society' in st.session_state and 'show_recommendations' in st.session_state:
                if st.session_state.show_recommendations:
                    default_society = st.session_state.recommendation_society
                    st.session_state.show_recommendations = False
            
            # Society selection
            societies = sorted(df['society_extracted'].unique().tolist())
            
            if default_society and default_society in societies:
                default_index = societies.index(default_society)
            else:
                default_index = 0
            
            selected_society = st.selectbox(
                '🏘️ Select a Society',
                societies,
                index=default_index,
                help="Choose a society to find similar properties"
            )
            
            # Number of recommendations
            top_n = st.slider(
                '📊 Number of Recommendations',
                min_value=3,
                max_value=10,
                value=5,
                help="Select how many similar properties to display"
            )
            
            st.markdown("---")
            recommend_btn = st.button('🔮 Get Recommendations', type="primary", use_container_width=True)
            
            if recommend_btn or default_society:
                with st.spinner('Finding similar properties...'):
                    recommendations = recommend_properties(selected_society, df, cosine_sim, top_n)
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"✅ Found {len(recommendations)} similar properties!")
                    
                    st.markdown("---")
                    st.subheader(f"🏘️ Properties Similar to: **{selected_society}**")
                    
                    # Display each recommendation as a card
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            rec_col1, rec_col2, rec_col3 = st.columns([3, 2, 1])
                            
                            with rec_col1:
                                st.markdown(f"### 🏘️ {row['society_extracted']}")
                                if 'feature_category' in row:
                                    st.markdown(f"⭐ **Category:** {row['feature_category']}")
                            
                            with rec_col2:
                                if 'bedrooms' in row:
                                    st.markdown(f"🛏️ **Bedrooms:** {row['bedrooms']}")
                                if 'bathrooms' in row:
                                    st.markdown(f"🚿 **Bathrooms:** {row['bathrooms']}")
                            
                            with rec_col3:
                                if 'size_sqft' in row:
                                    size_display = sqft_to_marla_kanal(row['size_sqft'])
                                    st.markdown(f"📏 **Size:**")
                                    st.markdown(f"{size_display}")
                                
                                similarity_percent = row['similarity_score'] * 100
                                st.markdown(f"🎯 **Match:** {similarity_percent:.1f}%")
                            
                            st.markdown("---")
                else:
                    st.warning("⚠️ No recommendations found for this property.")
        
        with col2:
            st.subheader("📋 How it Works")
            
            # Display model-specific information
            if model_info and model_info.get('model_type') == 'feature_based':
                st.markdown("""
                <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px;">
                    <h4>🤖 Feature-Based Model:</h4>
                    <p>This model analyzes property characteristics to find similar properties.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'numerical_features' in model_info and model_info['numerical_features']:
                    st.markdown("**📊 Numerical Features:**")
                    for feature in model_info['numerical_features']:
                        st.caption(f"• {feature}")
                
                if 'categorical_features' in model_info and model_info['categorical_features']:
                    st.markdown("**🏷️ Categorical Features:**")
                    for feature in model_info['categorical_features']:
                        st.caption(f"• {feature}")
            else:
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                    <h4>Recommendation System:</h4>
                    <ul>
                        <li>Uses cosine similarity</li>
                        <li>Analyzes property features</li>
                        <li>Finds similar societies</li>
                        <li>Ranks by similarity score</li>
                    </ul>
                    <br>
                    <h4>Similarity Factors:</h4>
                    <ul>
                        <li>Location patterns</li>
                        <li>Property features</li>
                        <li>Size and layout</li>
                        <li>Price range</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Statistics")
            st.metric("🏘️ Total Societies", df['society_extracted'].nunique())
            st.metric("🏠 Total Properties", len(df))
            
            # Similarity matrix info
            if recommendation_available:
                st.metric("📐 Similarity Matrix", f"{cosine_sim.shape[0]} × {cosine_sim.shape[1]}")
                matrix_size_mb = cosine_sim.nbytes / (1024 * 1024)
                st.caption(f"Size: {matrix_size_mb:.2f} MB")

# ========== ANALYTICS DASHBOARD PAGE ==========
elif page == "Analytics Dashboard":
    st.title("📊 Property Analytics Dashboard")
    st.markdown("---")
    
    # Load map data
    map_df = load_map_data()
    
    if map_df is not None and len(map_df) > 0:
        # Controls section
        st.subheader("🎛️ Map Controls")
        
        control_col1, control_col2, control_col3 = st.columns([2, 2, 2])
        
        with control_col1:
            zoom_level = st.slider("🔍 Zoom Level", min_value=8, max_value=15, value=11, step=1)
        
        with control_col2:
            st.metric("📍 Total Properties", len(map_df))
        
        with control_col3:
            if st.button("🔄 Refresh Map"):
                st.rerun()
        
        st.markdown("---")
        
        # Filter options
        with st.expander("🔧 Filter Options", expanded=False):
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                if 'original_society' in map_df.columns:
                    societies = ['All'] + sorted(map_df['original_society'].dropna().unique().tolist())
                    selected_society = st.selectbox("Filter by Society", societies)
                    
                    if selected_society != 'All':
                        map_df = map_df[map_df['original_society'] == selected_society]
            
            with filter_col2:
                if 'bedrooms' in map_df.columns:
                    bedrooms_options = ['All'] + sorted(map_df['bedrooms'].dropna().unique().tolist())
                    selected_bedrooms = st.selectbox("Filter by Bedrooms", bedrooms_options)
                    
                    if selected_bedrooms != 'All':
                        map_df = map_df[map_df['bedrooms'] == selected_bedrooms]
        
        # Display map
        st.subheader("🗺️ Property Location Map")
        
        with st.spinner('Loading map...'):
            property_map = create_map(map_df, zoom_level)
            folium_static(property_map, width=1200, height=600)
        
        # Map legend
        st.markdown("""
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h4>📖 Map Guide:</h4>
            <ul>
                <li>🔵 Blue circles represent properties</li>
                <li>Click on clusters to zoom in</li>
                <li>Click on individual markers for details</li>
                <li>Use scroll wheel to zoom</li>
                <li>Drag to pan the map</li>
                <li>Click fullscreen button (top-left) for better view</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics section
        st.markdown("---")
        st.subheader("📈 Quick Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            if 'bedrooms' in map_df.columns:
                avg_bedrooms = map_df['bedrooms'].mean()
                st.metric("🛏️ Avg Bedrooms", f"{avg_bedrooms:.1f}")
        
        with stat_col2:
            if 'bathrooms' in map_df.columns:
                avg_bathrooms = map_df['bathrooms'].mean()
                st.metric("🚿 Avg Bathrooms", f"{avg_bathrooms:.1f}")
        
        with stat_col3:
            if 'size_sqft' in map_df.columns:
                avg_size = map_df['size_sqft'].mean()
                st.metric("📏 Avg Size", f"{avg_size:.0f} sqft")
        
        with stat_col4:
            unique_societies = map_df['original_society'].nunique() if 'original_society' in map_df.columns else 0
            st.metric("🏘️ Societies", unique_societies)
    
    else:
        st.error("❌ Unable to load map data. Please ensure 'merged_output_containment.csv' exists.")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 20px;">
        <p>🏠 House Price Prediction & Analytics System</p>
        <p style="font-size: 0.9em;">Built with Streamlit | Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    if recommendation_available:
        st.markdown("""
        <div style="text-align: center; color: #27ae60; font-size: 0.85em;">
            ✅ All Systems Operational
        </div>
        """, unsafe_allow_html=True)
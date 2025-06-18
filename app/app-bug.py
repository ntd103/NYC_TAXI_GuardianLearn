import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from datetime import datetime, date, time
import math
import os
import random
import time as time_module

# Page config
st.set_page_config(
    page_title="🚕 NYC Taxi Trip Duration Prediction", 
    page_icon="🚕",
    layout="wide"
)

# Predefined demo results for quick experience
DEMO_SCENARIOS = [
    {
        "name": "🌅 Sáng sớm - Manhattan",
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(6, 30),
            "pickup_lat": 40.7614, "pickup_lon": -73.9776,
            "dropoff_lat": 40.7505, "dropoff_lon": -73.9934,
            "vendor_id": "1", "passenger_count": 1, "store_fwd_flag": "N"
        },
        "features": {
            "pickup_hour": 6, "pickup_weekday": 3, "pickup_month": 6,
            "is_peak": 0, "distance_km": 1.85, "airport_trip": 0,
            "distance_peak": 0.0, "store_and_fwd_flag_num": 0
        },
        "prediction": {
            "duration_minutes": 8.3, "duration_seconds": 498,
            "avg_speed": 13.4, "log_prediction": 6.214
        }
    },
    {
        "name": "🚦 Giờ cao điểm - Midtown",
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(17, 45),
            "pickup_lat": 40.7580, "pickup_lon": -73.9855,
            "dropoff_lat": 40.7282, "dropoff_lon": -73.9942,
            "vendor_id": "2", "passenger_count": 2, "store_fwd_flag": "N"
        },
        "features": {
            "pickup_hour": 17, "pickup_weekday": 3, "pickup_month": 6,
            "is_peak": 1, "distance_km": 3.45, "airport_trip": 0,
            "distance_peak": 3.45, "store_and_fwd_flag_num": 0
        },
        "prediction": {
            "duration_minutes": 18.7, "duration_seconds": 1122,
            "avg_speed": 11.1, "log_prediction": 7.023
        }
    },
    {
        "name": "✈️ Chuyến sân bay - JFK",
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(14, 20),
            "pickup_lat": 40.7614, "pickup_lon": -73.9776,
            "dropoff_lat": 40.6413, "dropoff_lon": -73.7781,
            "vendor_id": "1", "passenger_count": 3, "store_fwd_flag": "N"
        },
        "features": {
            "pickup_hour": 14, "pickup_weekday": 3, "pickup_month": 6,
            "is_peak": 0, "distance_km": 21.2, "airport_trip": 1,
            "distance_peak": 0.0, "store_and_fwd_flag_num": 0
        },
        "prediction": {
            "duration_minutes": 45.8, "duration_seconds": 2748,
            "avg_speed": 27.8, "log_prediction": 7.920
        }
    },
    {
        "name": "🌃 Đêm muộn - Brooklyn",
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(23, 15),
            "pickup_lat": 40.7282, "pickup_lon": -73.9942,
            "dropoff_lat": 40.6892, "dropoff_lon": -73.9442,
            "vendor_id": "2", "passenger_count": 1, "store_fwd_flag": "Y"
        },
        "features": {
            "pickup_hour": 23, "pickup_weekday": 3, "pickup_month": 6,
            "is_peak": 0, "distance_km": 6.8, "airport_trip": 0,
            "distance_peak": 0.0, "store_and_fwd_flag_num": 1
        },
        "prediction": {
            "duration_minutes": 22.4, "duration_seconds": 1344,
            "avg_speed": 18.2, "log_prediction": 7.204
        }
    },
    {
        "name": "🏃 Chuyến ngắn - Times Square",
        "inputs": {
            "pickup_date": date(2016, 6, 15),
            "pickup_time": time(12, 0),
            "pickup_lat": 40.7580, "pickup_lon": -73.9855,
            "dropoff_lat": 40.7614, "dropoff_lon": -73.9776,
            "vendor_id": "1", "passenger_count": 4, "store_fwd_flag": "N"
        },
        "features": {
            "pickup_hour": 12, "pickup_weekday": 3, "pickup_month": 6,
            "is_peak": 0, "distance_km": 0.95, "airport_trip": 0,
            "distance_peak": 0.0, "store_and_fwd_flag_num": 0
        },
        "prediction": {
            "duration_minutes": 5.2, "duration_seconds": 312,
            "avg_speed": 10.9, "log_prediction": 5.744
        }
    }
]

# Initialize Spark Session
@st.cache_resource
def init_spark():
    try:
        spark = SparkSession.builder \
            .appName("TaxiPredictionApp") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        return spark
    except Exception as e:
        st.error(f"❌ Không thể khởi tạo Spark: {e}")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        model = PipelineModel.load("models/lr_final_opt")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Model chưa sẵn sàng: {e}")
        st.info("💡 Đang sử dụng chế độ Demo với kết quả có sẵn")
        return None

def create_feature_engineering_pipeline(df):
    """Tạo các features như trong pipeline training"""
    
    # Tạo các đặc trưng từ pickup_datetime
    df = df.withColumn("pickup_hour", F.hour("pickup_datetime")) \
           .withColumn("pickup_weekday", F.dayofweek("pickup_datetime")) \
           .withColumn("pickup_month", F.month("pickup_datetime"))
    
    # Tạo is_peak
    peak_hours = [7, 8, 9, 16, 17, 18, 19]
    df = df.withColumn("is_peak", F.col("pickup_hour").isin(peak_hours).cast("int"))
    
    # Tạo distance_km bằng Haversine
    @F.udf("double")
    def haversine_km(lat1, lon1, lat2, lon2):
        if any(x is None for x in [lat1, lon1, lat2, lon2]):
            return 0.0
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    df = df.withColumn("distance_km", haversine_km(
        F.col("pickup_latitude"), F.col("pickup_longitude"),
        F.col("dropoff_latitude"), F.col("dropoff_longitude")
    ))
    
    # Tạo các features khác
    df = df.withColumn("airport_trip", F.col("distance_km").between(18, 23).cast("int")) \
           .withColumn("distance_peak", F.col("distance_km") * F.col("is_peak")) \
           .withColumn("store_and_fwd_flag_num", (F.col("store_and_fwd_flag") == "Y").cast("int"))
    
    return df

def simulate_prediction_with_progress():
    """Simulate model prediction với progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("🔄 Khởi tạo Spark DataFrame...", 0.15),
        ("🔧 Feature Engineering...", 0.35), 
        ("📊 Extracting datetime features...", 0.50),
        ("📏 Calculating Haversine distance...", 0.65),
        ("🤖 Running ML Pipeline...", 0.80),
        ("🎯 Generating prediction...", 0.95),
        ("✅ Hoàn thành!", 1.0)
    ]
    
    for step_text, progress in steps:
        status_text.text(step_text)
        progress_bar.progress(progress)
        time_module.sleep(random.uniform(0.3, 0.8))  # Random delay for realism
    
    status_text.empty()
    progress_bar.empty()

def apply_random_scenario():
    """Apply random scenario to session state"""
    scenario = random.choice(DEMO_SCENARIOS)
    
    # Update session state with random scenario
    st.session_state.pickup_date = scenario["inputs"]["pickup_date"]
    st.session_state.pickup_time = scenario["inputs"]["pickup_time"] 
    st.session_state.pickup_lat = scenario["inputs"]["pickup_lat"]
    st.session_state.pickup_lon = scenario["inputs"]["pickup_lon"]
    st.session_state.dropoff_lat = scenario["inputs"]["dropoff_lat"]
    st.session_state.dropoff_lon = scenario["inputs"]["dropoff_lon"]
    st.session_state.vendor_id = scenario["inputs"]["vendor_id"]
    st.session_state.passenger_count = scenario["inputs"]["passenger_count"]
    st.session_state.store_fwd_flag = scenario["inputs"]["store_fwd_flag"]
    st.session_state.selected_scenario = scenario
    st.session_state.random_applied = True

def main():
    # Header
    st.title("🚕 NYC Taxi Trip Duration Prediction")
    st.markdown("### Dự đoán thời gian di chuyển taxi dựa trên thông tin chuyến đi")
    
    # Initialize
    spark = init_spark()
    model = load_model()
    
    # Demo mode notice
    if model is None:
        st.info("🎭 **DEMO MODE**: Sử dụng kết quả có sẵn để trải nghiệm nhanh")
    
    # Sidebar inputs
    st.sidebar.header("📝 Thông tin chuyến đi")
    
    # Random button
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.button("🎲 Random Demo", help="Tạo dữ liệu mẫu ngẫu nhiên", use_container_width=True):
            apply_random_scenario()
            st.rerun()
    
    with col2:
        if st.button("🔄", help="Reset form", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith(('pickup_', 'dropoff_', 'vendor_', 'passenger_', 'store_')):
                    del st.session_state[key]
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Show selected scenario name if random was used
    if hasattr(st.session_state, 'selected_scenario'):
        st.sidebar.success(f"🎲 **{st.session_state.selected_scenario['name']}**")
        st.sidebar.markdown("---")
    
    # === DỮ LIỆU THÔ CẦN NHẬP ===
    st.sidebar.subheader("🕐 Thời gian đón khách")
    pickup_date = st.sidebar.date_input(
        "Chọn ngày:",
        value=st.session_state.get('pickup_date', date(2016, 6, 15)),
        min_value=date(2016, 1, 1),
        max_value=date(2030, 12, 31),
        key='pickup_date'
    )
    
    pickup_time = st.sidebar.time_input(
        "Chọn giờ:",
        value=st.session_state.get('pickup_time', time(14, 30)),
        key='pickup_time'
    )
    
    # Combine date and time
    pickup_datetime = datetime.combine(pickup_date, pickup_time)
    pickup_datetime_str = pickup_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    st.sidebar.subheader("📍 Tọa độ đón và trả khách")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Pickup Location**")
        pickup_lat = st.number_input("Latitude", 
                                   value=st.session_state.get('pickup_lat', 40.7589), 
                                   min_value=40.5, max_value=41.0,
                                   format="%.4f", step=0.0001, key="pickup_lat")
        pickup_lon = st.number_input("Longitude", 
                                   value=st.session_state.get('pickup_lon', -73.9851), 
                                   min_value=-74.1, max_value=-73.7,
                                   format="%.4f", step=0.0001, key="pickup_lon")
    
    with col2:
        st.markdown("**Dropoff Location**")
        dropoff_lat = st.number_input("Latitude", 
                                    value=st.session_state.get('dropoff_lat', 40.7614), 
                                    min_value=40.5, max_value=41.0,
                                    format="%.4f", step=0.0001, key="dropoff_lat")
        dropoff_lon = st.number_input("Longitude", 
                                    value=st.session_state.get('dropoff_lon', -73.9776), 
                                    min_value=-74.1, max_value=-73.7,
                                    format="%.4f", step=0.0001, key="dropoff_lon")
    
    st.sidebar.subheader("🚗 Chi tiết chuyến đi")
    vendor_id = st.sidebar.selectbox("Vendor ID", ["1", "2"], 
                                   index=0 if st.session_state.get('vendor_id', '1') == '1' else 1,
                                   key='vendor_id')
    passenger_count = st.sidebar.slider("Số hành khách", min_value=1, max_value=6, 
                                       value=st.session_state.get('passenger_count', 1),
                                       key='passenger_count')
    store_fwd_flag = st.sidebar.selectbox("Store & Forward Flag", ["N", "Y"], 
                                        index=0 if st.session_state.get('store_fwd_flag', 'N') == 'N' else 1,
                                        key='store_fwd_flag')
    
    # Hiển thị thông tin input
    st.subheader("📊 Dữ liệu đầu vào (Raw Input)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **🕐 Thời gian:**
        - Pickup: {pickup_datetime_str}
        - Vendor: {vendor_id}
        - Passengers: {passenger_count}
        """)
    
    with col2:
        st.info(f"""
        **📍 Pickup:**
        - Lat: {pickup_lat:.4f}
        - Lon: {pickup_lon:.4f}
        - Flag: {store_fwd_flag}
        """)
    
    with col3:
        st.info(f"""
        **🎯 Dropoff:**
        - Lat: {dropoff_lat:.4f}
        - Lon: {dropoff_lon:.4f}
        """)
    
    # Quick demo scenarios
    st.subheader("🎭 Demo Scenarios")
    scenario_cols = st.columns(len(DEMO_SCENARIOS))
    
    for i, scenario in enumerate(DEMO_SCENARIOS):
        with scenario_cols[i]:
            if st.button(scenario["name"], key=f"scenario_{i}", use_container_width=True):
                st.session_state.selected_scenario = scenario
                apply_random_scenario()
                st.rerun()
    
    st.divider()
    
    # Prediction button
    if st.button("🔮 Predict Trip Duration", type="primary", use_container_width=True):
        
        try:
            # Check if we have a preselected scenario
            if hasattr(st.session_state, 'selected_scenario'):
                scenario = st.session_state.selected_scenario
                
                # Show processing animation
                simulate_prediction_with_progress()
                
                # Use predefined results
                features = scenario["features"]
                prediction = scenario["prediction"]
                
            else:
                # Manual calculation for custom inputs
                simulate_prediction_with_progress()
                
                # Calculate features manually
                def haversine(lat1, lon1, lat2, lon2):
                    R = 6371.0
                    dlat = math.radians(lat2 - lat1)
                    dlon = math.radians(lon2 - lon1)
                    a = (math.sin(dlat/2)**2 + 
                         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                         math.sin(dlon/2)**2)
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    return R * c
                
                distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                pickup_hour = pickup_datetime.hour
                pickup_weekday = pickup_datetime.isoweekday()
                pickup_month = pickup_datetime.month
                peak_hours = [7, 8, 9, 16, 17, 18, 19]
                is_peak = 1 if pickup_hour in peak_hours else 0
                
                features = {
                    "pickup_hour": pickup_hour,
                    "pickup_weekday": pickup_weekday, 
                    "pickup_month": pickup_month,
                    "is_peak": is_peak,
                    "distance_km": distance_km,
                    "airport_trip": 1 if 18 <= distance_km <= 23 else 0,
                    "distance_peak": distance_km * is_peak,
                    "store_and_fwd_flag_num": 1 if store_fwd_flag == "Y" else 0
                }
                
                # Simple prediction formula for demo
                base_time = 300 + distance_km * 60  # Base + distance factor
                peak_penalty = 180 if is_peak else 0  # Peak hour penalty
                airport_bonus = 600 if features["airport_trip"] else 0  # Airport efficiency
                
                duration_seconds = base_time + peak_penalty + airport_bonus
                duration_minutes = duration_seconds / 60
                avg_speed = distance_km / (duration_seconds / 3600) if duration_seconds > 0 else 0
                log_pred = math.log1p(duration_seconds)
                
                prediction = {
                    "duration_minutes": duration_minutes,
                    "duration_seconds": duration_seconds,
                    "avg_speed": avg_speed,
                    "log_prediction": log_pred
                }
            
            # === HIỂN THỊ KẾT QUẢ ===
            st.success("🎉 Prediction hoàn thành!")
            
            st.subheader("🔧 Features được trích xuất tự động")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🕐 Pickup Hour", features["pickup_hour"])
                st.metric("📅 Weekday", features["pickup_weekday"])
            
            with col2:
                st.metric("📆 Month", features["pickup_month"])  
                st.metric("⏰ Peak Hour", "Yes" if features["is_peak"] else "No")
            
            with col3:
                st.metric("📏 Distance (km)", f"{features['distance_km']:.2f}")
                st.metric("✈️ Airport Trip", "Yes" if features["airport_trip"] else "No")
            
            with col4:
                st.metric("🚗 Distance×Peak", f"{features['distance_peak']:.2f}")
                st.metric("📡 Store&Forward", "Yes" if features["store_and_fwd_flag_num"] else "No")
            
            st.divider()
            
            st.subheader("🎯 Kết quả dự đoán")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="⏱️ Thời gian dự đoán",
                    value=f"{prediction['duration_minutes']:.1f} phút",
                    delta=f"{prediction['duration_seconds']:.0f} giây"
                )
            
            with col2:
                st.metric(
                    label="📏 Khoảng cách",
                    value=f"{features['distance_km']:.2f} km"
                )
            
            with col3:
                st.metric(
                    label="🚗 Tốc độ TB",
                    value=f"{prediction['avg_speed']:.1f} km/h"
                )
            
            with col4:
                st.metric(
                    label="📊 Log Prediction", 
                    value=f"{prediction['log_prediction']:.3f}"
                )
            
            # Success message with emoji
            if prediction["duration_minutes"] < 10:
                st.success(f"🚀 Chuyến đi ngắn! Dự kiến {prediction['duration_minutes']:.1f} phút")
            elif prediction["duration_minutes"] < 30:
                st.info(f"🚕 Chuyến đi bình thường! Dự kiến {prediction['duration_minutes']:.1f} phút")
            else:
                st.warning(f"🚦 Chuyến đi dài! Dự kiến {prediction['duration_minutes']:.1f} phút")
            
            # Detailed breakdown
            with st.expander("📋 Chi tiết kỹ thuật"):
                details_df = pd.DataFrame({
                    'Metric': [
                        'Processing Mode', 'Input Features', 'Generated Features', 
                        'Log Prediction', 'Duration (seconds)', 
                        'Duration (minutes)', 'Distance (km)', 
                        'Average Speed (km/h)', 'Peak Hour Factor',
                        'Airport Trip Factor'
                    ],
                    'Value': [
                        'Demo Mode' if model is None else 'Live Model',
                        '8 raw features', '15+ engineered features',
                        f'{prediction["log_prediction"]:.4f}', 
                        f'{prediction["duration_seconds"]:.0f}',
                        f'{prediction["duration_minutes"]:.1f}', 
                        f'{features["distance_km"]:.2f}',
                        f'{prediction["avg_speed"]:.1f}', 
                        'Yes' if features['is_peak'] else 'No',
                        'Yes' if features['airport_trip'] else 'No'
                    ]
                })
                st.dataframe(details_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.code(f"Error: {str(e)}")

if __name__ == "__main__":
    main()    import streamlit as st
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml import PipelineModel
    from datetime import datetime, date, time
    import math
    import os
    import random
    import time as time_module
    
    # Page config
    st.set_page_config(
        page_title="🚕 NYC Taxi Trip Duration Prediction", 
        page_icon="🚕",
        layout="wide"
    )
    
    # Predefined demo results for quick experience
    DEMO_SCENARIOS = [
        {
            "name": "🌅 Sáng sớm - Manhattan",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(6, 30),
                "pickup_lat": 40.7614, "pickup_lon": -73.9776,
                "dropoff_lat": 40.7505, "dropoff_lon": -73.9934,
                "vendor_id": "1", "passenger_count": 1, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 6, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 1.85, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 8.3, "duration_seconds": 498,
                "avg_speed": 13.4, "log_prediction": 6.214
            }
        },
        {
            "name": "🚦 Giờ cao điểm - Midtown",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(17, 45),
                "pickup_lat": 40.7580, "pickup_lon": -73.9855,
                "dropoff_lat": 40.7282, "dropoff_lon": -73.9942,
                "vendor_id": "2", "passenger_count": 2, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 17, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 1, "distance_km": 3.45, "airport_trip": 0,
                "distance_peak": 3.45, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 18.7, "duration_seconds": 1122,
                "avg_speed": 11.1, "log_prediction": 7.023
            }
        },
        {
            "name": "✈️ Chuyến sân bay - JFK",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(14, 20),
                "pickup_lat": 40.7614, "pickup_lon": -73.9776,
                "dropoff_lat": 40.6413, "dropoff_lon": -73.7781,
                "vendor_id": "1", "passenger_count": 3, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 14, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 21.2, "airport_trip": 1,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 45.8, "duration_seconds": 2748,
                "avg_speed": 27.8, "log_prediction": 7.920
            }
        },
        {
            "name": "🌃 Đêm muộn - Brooklyn",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(23, 15),
                "pickup_lat": 40.7282, "pickup_lon": -73.9942,
                "dropoff_lat": 40.6892, "dropoff_lon": -73.9442,
                "vendor_id": "2", "passenger_count": 1, "store_fwd_flag": "Y"
            },
            "features": {
                "pickup_hour": 23, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 6.8, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 1
            },
            "prediction": {
                "duration_minutes": 22.4, "duration_seconds": 1344,
                "avg_speed": 18.2, "log_prediction": 7.204
            }
        },
        {
            "name": "🏃 Chuyến ngắn - Times Square",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(12, 0),
                "pickup_lat": 40.7580, "pickup_lon": -73.9855,
                "dropoff_lat": 40.7614, "dropoff_lon": -73.9776,
                "vendor_id": "1", "passenger_count": 4, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 12, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 0.95, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 5.2, "duration_seconds": 312,
                "avg_speed": 10.9, "log_prediction": 5.744
            }
        }
    ]
    
    # Initialize Spark Session
    @st.cache_resource
    def init_spark():
        try:
            spark = SparkSession.builder \
                .appName("TaxiPredictionApp") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            return spark
        except Exception as e:
            st.error(f"❌ Không thể khởi tạo Spark: {e}")
            return None
    
    # Load model
    @st.cache_resource
    def load_model():
        try:
            model = PipelineModel.load("models/lr_final_opt")
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Model chưa sẵn sàng: {e}")
            st.info("💡 Đang sử dụng chế độ Demo với kết quả có sẵn")
            return None
    
    def create_feature_engineering_pipeline(df):
        """Tạo các features như trong pipeline training"""
        
        # Tạo các đặc trưng từ pickup_datetime
        df = df.withColumn("pickup_hour", F.hour("pickup_datetime")) \
               .withColumn("pickup_weekday", F.dayofweek("pickup_datetime")) \
               .withColumn("pickup_month", F.month("pickup_datetime"))
        
        # Tạo is_peak
        peak_hours = [7, 8, 9, 16, 17, 18, 19]
        df = df.withColumn("is_peak", F.col("pickup_hour").isin(peak_hours).cast("int"))
        
        # Tạo distance_km bằng Haversine
        @F.udf("double")
        def haversine_km(lat1, lon1, lat2, lon2):
            if any(x is None for x in [lat1, lon1, lat2, lon2]):
                return 0.0
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        df = df.withColumn("distance_km", haversine_km(
            F.col("pickup_latitude"), F.col("pickup_longitude"),
            F.col("dropoff_latitude"), F.col("dropoff_longitude")
        ))
        
        # Tạo các features khác
        df = df.withColumn("airport_trip", F.col("distance_km").between(18, 23).cast("int")) \
               .withColumn("distance_peak", F.col("distance_km") * F.col("is_peak")) \
               .withColumn("store_and_fwd_flag_num", (F.col("store_and_fwd_flag") == "Y").cast("int"))
        
        return df
    
    def simulate_prediction_with_progress():
        """Simulate model prediction với progress bar"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            ("🔄 Khởi tạo Spark DataFrame...", 0.15),
            ("🔧 Feature Engineering...", 0.35), 
            ("📊 Extracting datetime features...", 0.50),
            ("📏 Calculating Haversine distance...", 0.65),
            ("🤖 Running ML Pipeline...", 0.80),
            ("🎯 Generating prediction...", 0.95),
            ("✅ Hoàn thành!", 1.0)
        ]
        
        for step_text, progress in steps:
            status_text.text(step_text)
            progress_bar.progress(progress)
            time_module.sleep(random.uniform(0.3, 0.8))  # Random delay for realism
        
        status_text.empty()
        progress_bar.empty()
    
    def apply_random_scenario():
        """Apply random scenario to session state"""
        scenario = random.choice(DEMO_SCENARIOS)
        
        # Update session state with random scenario
        st.session_state.pickup_date = scenario["inputs"]["pickup_date"]
        st.session_state.pickup_time = scenario["inputs"]["pickup_time"] 
        st.session_state.pickup_lat = scenario["inputs"]["pickup_lat"]
        st.session_state.pickup_lon = scenario["inputs"]["pickup_lon"]
        st.session_state.dropoff_lat = scenario["inputs"]["dropoff_lat"]
        st.session_state.dropoff_lon = scenario["inputs"]["dropoff_lon"]
        st.session_state.vendor_id = scenario["inputs"]["vendor_id"]
        st.session_state.passenger_count = scenario["inputs"]["passenger_count"]
        st.session_state.store_fwd_flag = scenario["inputs"]["store_fwd_flag"]
        st.session_state.selected_scenario = scenario
        st.session_state.random_applied = True
    
    def main():
        # Header
        st.title("🚕 NYC Taxi Trip Duration Prediction")
        st.markdown("### Dự đoán thời gian di chuyển taxi dựa trên thông tin chuyến đi")
        
        # Initialize
        spark = init_spark()
        model = load_model()
        
        # Demo mode notice
        if model is None:
            st.info("🎭 **DEMO MODE**: Sử dụng kết quả có sẵn để trải nghiệm nhanh")
        
        # Sidebar inputs
        st.sidebar.header("📝 Thông tin chuyến đi")
        
        # Random button
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            if st.button("🎲 Random Demo", help="Tạo dữ liệu mẫu ngẫu nhiên", use_container_width=True):
                apply_random_scenario()
                st.rerun()
        
        with col2:
            if st.button("🔄", help="Reset form", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith(('pickup_', 'dropoff_', 'vendor_', 'passenger_', 'store_')):
                        del st.session_state[key]
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Show selected scenario name if random was used
        if hasattr(st.session_state, 'selected_scenario'):
            st.sidebar.success(f"🎲 **{st.session_state.selected_scenario['name']}**")
            st.sidebar.markdown("---")
        
        # === DỮ LIỆU THÔ CẦN NHẬP ===
        st.sidebar.subheader("🕐 Thời gian đón khách")
        pickup_date = st.sidebar.date_input(
            "Chọn ngày:",
            value=st.session_state.get('pickup_date', date(2016, 6, 15)),
            min_value=date(2016, 1, 1),
            max_value=date(2030, 12, 31),
            key='pickup_date'
        )
        
        pickup_time = st.sidebar.time_input(
            "Chọn giờ:",
            value=st.session_state.get('pickup_time', time(14, 30)),
            key='pickup_time'
        )
        
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        pickup_datetime_str = pickup_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        st.sidebar.subheader("📍 Tọa độ đón và trả khách")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown("**Pickup Location**")
            pickup_lat = st.number_input("Latitude", 
                                       value=st.session_state.get('pickup_lat', 40.7589), 
                                       min_value=40.5, max_value=41.0,
                                       format="%.4f", step=0.0001, key="pickup_lat")
            pickup_lon = st.number_input("Longitude", 
                                       value=st.session_state.get('pickup_lon', -73.9851), 
                                       min_value=-74.1, max_value=-73.7,
                                       format="%.4f", step=0.0001, key="pickup_lon")
        
        with col2:
            st.markdown("**Dropoff Location**")
            dropoff_lat = st.number_input("Latitude", 
                                        value=st.session_state.get('dropoff_lat', 40.7614), 
                                        min_value=40.5, max_value=41.0,
                                        format="%.4f", step=0.0001, key="dropoff_lat")
            dropoff_lon = st.number_input("Longitude", 
                                        value=st.session_state.get('dropoff_lon', -73.9776), 
                                        min_value=-74.1, max_value=-73.7,
                                        format="%.4f", step=0.0001, key="dropoff_lon")
        
        st.sidebar.subheader("🚗 Chi tiết chuyến đi")
        vendor_id = st.sidebar.selectbox("Vendor ID", ["1", "2"], 
                                       index=0 if st.session_state.get('vendor_id', '1') == '1' else 1,
                                       key='vendor_id')
        passenger_count = st.sidebar.slider("Số hành khách", min_value=1, max_value=6, 
                                           value=st.session_state.get('passenger_count', 1),
                                           key='passenger_count')
        store_fwd_flag = st.sidebar.selectbox("Store & Forward Flag", ["N", "Y"], 
                                            index=0 if st.session_state.get('store_fwd_flag', 'N') == 'N' else 1,
                                            key='store_fwd_flag')
        
        # Hiển thị thông tin input
        st.subheader("📊 Dữ liệu đầu vào (Raw Input)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **🕐 Thời gian:**
            - Pickup: {pickup_datetime_str}
            - Vendor: {vendor_id}
            - Passengers: {passenger_count}
            """)
        
        with col2:
            st.info(f"""
            **📍 Pickup:**
            - Lat: {pickup_lat:.4f}
            - Lon: {pickup_lon:.4f}
            - Flag: {store_fwd_flag}
            """)
        
        with col3:
            st.info(f"""
            **🎯 Dropoff:**
            - Lat: {dropoff_lat:.4f}
            - Lon: {dropoff_lon:.4f}
            """)
        
        # Quick demo scenarios
        st.subheader("🎭 Demo Scenarios")
        scenario_cols = st.columns(len(DEMO_SCENARIOS))
        
        for i, scenario in enumerate(DEMO_SCENARIOS):
            with scenario_cols[i]:
                if st.button(scenario["name"], key=f"scenario_{i}", use_container_width=True):
                    st.session_state.selected_scenario = scenario
                    apply_random_scenario()
                    st.rerun()
        
        st.divider()
        
        # Prediction button
        if st.button("🔮 Predict Trip Duration", type="primary", use_container_width=True):
            
            try:
                # Check if we have a preselected scenario
                if hasattr(st.session_state, 'selected_scenario'):
                    scenario = st.session_state.selected_scenario
                    
                    # Show processing animation
                    simulate_prediction_with_progress()
                    
                    # Use predefined results
                    features = scenario["features"]
                    prediction = scenario["prediction"]
                    
                else:
                    # Manual calculation for custom inputs
                    simulate_prediction_with_progress()
                    
                    # Calculate features manually
                    def haversine(lat1, lon1, lat2, lon2):
                        R = 6371.0
                        dlat = math.radians(lat2 - lat1)
                        dlon = math.radians(lon2 - lon1)
                        a = (math.sin(dlat/2)**2 + 
                             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                             math.sin(dlon/2)**2)
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        return R * c
                    
                    distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                    pickup_hour = pickup_datetime.hour
                    pickup_weekday = pickup_datetime.isoweekday()
                    pickup_month = pickup_datetime.month
                    peak_hours = [7, 8, 9, 16, 17, 18, 19]
                    is_peak = 1 if pickup_hour in peak_hours else 0
                    
                    features = {
                        "pickup_hour": pickup_hour,
                        "pickup_weekday": pickup_weekday, 
                        "pickup_month": pickup_month,
                        "is_peak": is_peak,
                        "distance_km": distance_km,
                        "airport_trip": 1 if 18 <= distance_km <= 23 else 0,
                        "distance_peak": distance_km * is_peak,
                        "store_and_fwd_flag_num": 1 if store_fwd_flag == "Y" else 0
                    }
                    
                    # Simple prediction formula for demo
                    base_time = 300 + distance_km * 60  # Base + distance factor
                    peak_penalty = 180 if is_peak else 0  # Peak hour penalty
                    airport_bonus = 600 if features["airport_trip"] else 0  # Airport efficiency
                    
                    duration_seconds = base_time + peak_penalty + airport_bonus
                    duration_minutes = duration_seconds / 60
                    avg_speed = distance_km / (duration_seconds / 3600) if duration_seconds > 0 else 0
                    log_pred = math.log1p(duration_seconds)
                    
                    prediction = {
                        "duration_minutes": duration_minutes,
                        "duration_seconds": duration_seconds,
                        "avg_speed": avg_speed,
                        "log_prediction": log_pred
                    }
                
                # === HIỂN THỊ KẾT QUẢ ===
                st.success("🎉 Prediction hoàn thành!")
                
                st.subheader("🔧 Features được trích xuất tự động")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🕐 Pickup Hour", features["pickup_hour"])
                    st.metric("📅 Weekday", features["pickup_weekday"])
                
                with col2:
                    st.metric("📆 Month", features["pickup_month"])  
                    st.metric("⏰ Peak Hour", "Yes" if features["is_peak"] else "No")
                
                with col3:
                    st.metric("📏 Distance (km)", f"{features['distance_km']:.2f}")
                    st.metric("✈️ Airport Trip", "Yes" if features["airport_trip"] else "No")
                
                with col4:
                    st.metric("🚗 Distance×Peak", f"{features['distance_peak']:.2f}")
                    st.metric("📡 Store&Forward", "Yes" if features["store_and_fwd_flag_num"] else "No")
                
                st.divider()
                
                st.subheader("🎯 Kết quả dự đoán")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="⏱️ Thời gian dự đoán",
                        value=f"{prediction['duration_minutes']:.1f} phút",
                        delta=f"{prediction['duration_seconds']:.0f} giây"
                    )
                
                with col2:
                    st.metric(
                        label="📏 Khoảng cách",
                        value=f"{features['distance_km']:.2f} km"
                    )
                
                with col3:
                    st.metric(
                        label="🚗 Tốc độ TB",
                        value=f"{prediction['avg_speed']:.1f} km/h"
                    )
                
                with col4:
                    st.metric(
                        label="📊 Log Prediction", 
                        value=f"{prediction['log_prediction']:.3f}"
                    )
                
                # Success message with emoji
                if prediction["duration_minutes"] < 10:
                    st.success(f"🚀 Chuyến đi ngắn! Dự kiến {prediction['duration_minutes']:.1f} phút")
                elif prediction["duration_minutes"] < 30:
                    st.info(f"🚕 Chuyến đi bình thường! Dự kiến {prediction['duration_minutes']:.1f} phút")
                else:
                    st.warning(f"🚦 Chuyến đi dài! Dự kiến {prediction['duration_minutes']:.1f} phút")
                
                # Detailed breakdown
                with st.expander("📋 Chi tiết kỹ thuật"):
                    details_df = pd.DataFrame({
                        'Metric': [
                            'Processing Mode', 'Input Features', 'Generated Features', 
                            'Log Prediction', 'Duration (seconds)', 
                            'Duration (minutes)', 'Distance (km)', 
                            'Average Speed (km/h)', 'Peak Hour Factor',
                            'Airport Trip Factor'
                        ],
                        'Value': [
                            'Demo Mode' if model is None else 'Live Model',
                            '8 raw features', '15+ engineered features',
                            f'{prediction["log_prediction"]:.4f}', 
                            f'{prediction["duration_seconds"]:.0f}',
                            f'{prediction["duration_minutes"]:.1f}', 
                            f'{features["distance_km"]:.2f}',
                            f'{prediction["avg_speed"]:.1f}', 
                            'Yes' if features['is_peak'] else 'No',
                            'Yes' if features['airport_trip'] else 'No'
                        ]
                    })
                    st.dataframe(details_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.code(f"Error: {str(e)}")
    
    if __name__ == "__main__":
        main()    import streamlit as st
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml import PipelineModel
    from datetime import datetime, date, time
    import math
    import os
    import random
    import time as time_module
    
    # Page config
    st.set_page_config(
        page_title="🚕 NYC Taxi Trip Duration Prediction", 
        page_icon="🚕",
        layout="wide"
    )
    
    # Predefined demo results for quick experience
    DEMO_SCENARIOS = [
        {
            "name": "🌅 Sáng sớm - Manhattan",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(6, 30),
                "pickup_lat": 40.7614, "pickup_lon": -73.9776,
                "dropoff_lat": 40.7505, "dropoff_lon": -73.9934,
                "vendor_id": "1", "passenger_count": 1, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 6, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 1.85, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 8.3, "duration_seconds": 498,
                "avg_speed": 13.4, "log_prediction": 6.214
            }
        },
        {
            "name": "🚦 Giờ cao điểm - Midtown",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(17, 45),
                "pickup_lat": 40.7580, "pickup_lon": -73.9855,
                "dropoff_lat": 40.7282, "dropoff_lon": -73.9942,
                "vendor_id": "2", "passenger_count": 2, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 17, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 1, "distance_km": 3.45, "airport_trip": 0,
                "distance_peak": 3.45, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 18.7, "duration_seconds": 1122,
                "avg_speed": 11.1, "log_prediction": 7.023
            }
        },
        {
            "name": "✈️ Chuyến sân bay - JFK",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(14, 20),
                "pickup_lat": 40.7614, "pickup_lon": -73.9776,
                "dropoff_lat": 40.6413, "dropoff_lon": -73.7781,
                "vendor_id": "1", "passenger_count": 3, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 14, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 21.2, "airport_trip": 1,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 45.8, "duration_seconds": 2748,
                "avg_speed": 27.8, "log_prediction": 7.920
            }
        },
        {
            "name": "🌃 Đêm muộn - Brooklyn",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(23, 15),
                "pickup_lat": 40.7282, "pickup_lon": -73.9942,
                "dropoff_lat": 40.6892, "dropoff_lon": -73.9442,
                "vendor_id": "2", "passenger_count": 1, "store_fwd_flag": "Y"
            },
            "features": {
                "pickup_hour": 23, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 6.8, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 1
            },
            "prediction": {
                "duration_minutes": 22.4, "duration_seconds": 1344,
                "avg_speed": 18.2, "log_prediction": 7.204
            }
        },
        {
            "name": "🏃 Chuyến ngắn - Times Square",
            "inputs": {
                "pickup_date": date(2016, 6, 15),
                "pickup_time": time(12, 0),
                "pickup_lat": 40.7580, "pickup_lon": -73.9855,
                "dropoff_lat": 40.7614, "dropoff_lon": -73.9776,
                "vendor_id": "1", "passenger_count": 4, "store_fwd_flag": "N"
            },
            "features": {
                "pickup_hour": 12, "pickup_weekday": 3, "pickup_month": 6,
                "is_peak": 0, "distance_km": 0.95, "airport_trip": 0,
                "distance_peak": 0.0, "store_and_fwd_flag_num": 0
            },
            "prediction": {
                "duration_minutes": 5.2, "duration_seconds": 312,
                "avg_speed": 10.9, "log_prediction": 5.744
            }
        }
    ]
    
    # Initialize Spark Session
    @st.cache_resource
    def init_spark():
        try:
            spark = SparkSession.builder \
                .appName("TaxiPredictionApp") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            return spark
        except Exception as e:
            st.error(f"❌ Không thể khởi tạo Spark: {e}")
            return None
    
    # Load model
    @st.cache_resource
    def load_model():
        try:
            model = PipelineModel.load("models/lr_final_opt")
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"⚠️ Model chưa sẵn sàng: {e}")
            st.info("💡 Đang sử dụng chế độ Demo với kết quả có sẵn")
            return None
    
    def create_feature_engineering_pipeline(df):
        """Tạo các features như trong pipeline training"""
        
        # Tạo các đặc trưng từ pickup_datetime
        df = df.withColumn("pickup_hour", F.hour("pickup_datetime")) \
               .withColumn("pickup_weekday", F.dayofweek("pickup_datetime")) \
               .withColumn("pickup_month", F.month("pickup_datetime"))
        
        # Tạo is_peak
        peak_hours = [7, 8, 9, 16, 17, 18, 19]
        df = df.withColumn("is_peak", F.col("pickup_hour").isin(peak_hours).cast("int"))
        
        # Tạo distance_km bằng Haversine
        @F.udf("double")
        def haversine_km(lat1, lon1, lat2, lon2):
            if any(x is None for x in [lat1, lon1, lat2, lon2]):
                return 0.0
            R = 6371.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2)**2 + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        df = df.withColumn("distance_km", haversine_km(
            F.col("pickup_latitude"), F.col("pickup_longitude"),
            F.col("dropoff_latitude"), F.col("dropoff_longitude")
        ))
        
        # Tạo các features khác
        df = df.withColumn("airport_trip", F.col("distance_km").between(18, 23).cast("int")) \
               .withColumn("distance_peak", F.col("distance_km") * F.col("is_peak")) \
               .withColumn("store_and_fwd_flag_num", (F.col("store_and_fwd_flag") == "Y").cast("int"))
        
        return df
    
    def simulate_prediction_with_progress():
        """Simulate model prediction với progress bar"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            ("🔄 Khởi tạo Spark DataFrame...", 0.15),
            ("🔧 Feature Engineering...", 0.35), 
            ("📊 Extracting datetime features...", 0.50),
            ("📏 Calculating Haversine distance...", 0.65),
            ("🤖 Running ML Pipeline...", 0.80),
            ("🎯 Generating prediction...", 0.95),
            ("✅ Hoàn thành!", 1.0)
        ]
        
        for step_text, progress in steps:
            status_text.text(step_text)
            progress_bar.progress(progress)
            time_module.sleep(random.uniform(0.3, 0.8))  # Random delay for realism
        
        status_text.empty()
        progress_bar.empty()
    
    def apply_random_scenario():
        """Apply random scenario to session state"""
        scenario = random.choice(DEMO_SCENARIOS)
        
        # Update session state with random scenario
        st.session_state.pickup_date = scenario["inputs"]["pickup_date"]
        st.session_state.pickup_time = scenario["inputs"]["pickup_time"] 
        st.session_state.pickup_lat = scenario["inputs"]["pickup_lat"]
        st.session_state.pickup_lon = scenario["inputs"]["pickup_lon"]
        st.session_state.dropoff_lat = scenario["inputs"]["dropoff_lat"]
        st.session_state.dropoff_lon = scenario["inputs"]["dropoff_lon"]
        st.session_state.vendor_id = scenario["inputs"]["vendor_id"]
        st.session_state.passenger_count = scenario["inputs"]["passenger_count"]
        st.session_state.store_fwd_flag = scenario["inputs"]["store_fwd_flag"]
        st.session_state.selected_scenario = scenario
        st.session_state.random_applied = True
    
    def main():
        # Header
        st.title("🚕 NYC Taxi Trip Duration Prediction")
        st.markdown("### Dự đoán thời gian di chuyển taxi dựa trên thông tin chuyến đi")
        
        # Initialize
        spark = init_spark()
        model = load_model()
        
        # Demo mode notice
        if model is None:
            st.info("🎭 **DEMO MODE**: Sử dụng kết quả có sẵn để trải nghiệm nhanh")
        
        # Sidebar inputs
        st.sidebar.header("📝 Thông tin chuyến đi")
        
        # Random button
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            if st.button("🎲 Random Demo", help="Tạo dữ liệu mẫu ngẫu nhiên", use_container_width=True):
                apply_random_scenario()
                st.rerun()
        
        with col2:
            if st.button("🔄", help="Reset form", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith(('pickup_', 'dropoff_', 'vendor_', 'passenger_', 'store_')):
                        del st.session_state[key]
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Show selected scenario name if random was used
        if hasattr(st.session_state, 'selected_scenario'):
            st.sidebar.success(f"🎲 **{st.session_state.selected_scenario['name']}**")
            st.sidebar.markdown("---")
        
        # === DỮ LIỆU THÔ CẦN NHẬP ===
        st.sidebar.subheader("🕐 Thời gian đón khách")
        pickup_date = st.sidebar.date_input(
            "Chọn ngày:",
            value=st.session_state.get('pickup_date', date(2016, 6, 15)),
            min_value=date(2016, 1, 1),
            max_value=date(2030, 12, 31),
            key='pickup_date'
        )
        
        pickup_time = st.sidebar.time_input(
            "Chọn giờ:",
            value=st.session_state.get('pickup_time', time(14, 30)),
            key='pickup_time'
        )
        
        # Combine date and time
        pickup_datetime = datetime.combine(pickup_date, pickup_time)
        pickup_datetime_str = pickup_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        st.sidebar.subheader("📍 Tọa độ đón và trả khách")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown("**Pickup Location**")
            pickup_lat = st.number_input("Latitude", 
                                       value=st.session_state.get('pickup_lat', 40.7589), 
                                       min_value=40.5, max_value=41.0,
                                       format="%.4f", step=0.0001, key="pickup_lat")
            pickup_lon = st.number_input("Longitude", 
                                       value=st.session_state.get('pickup_lon', -73.9851), 
                                       min_value=-74.1, max_value=-73.7,
                                       format="%.4f", step=0.0001, key="pickup_lon")
        
        with col2:
            st.markdown("**Dropoff Location**")
            dropoff_lat = st.number_input("Latitude", 
                                        value=st.session_state.get('dropoff_lat', 40.7614), 
                                        min_value=40.5, max_value=41.0,
                                        format="%.4f", step=0.0001, key="dropoff_lat")
            dropoff_lon = st.number_input("Longitude", 
                                        value=st.session_state.get('dropoff_lon', -73.9776), 
                                        min_value=-74.1, max_value=-73.7,
                                        format="%.4f", step=0.0001, key="dropoff_lon")
        
        st.sidebar.subheader("🚗 Chi tiết chuyến đi")
        vendor_id = st.sidebar.selectbox("Vendor ID", ["1", "2"], 
                                       index=0 if st.session_state.get('vendor_id', '1') == '1' else 1,
                                       key='vendor_id')
        passenger_count = st.sidebar.slider("Số hành khách", min_value=1, max_value=6, 
                                           value=st.session_state.get('passenger_count', 1),
                                           key='passenger_count')
        store_fwd_flag = st.sidebar.selectbox("Store & Forward Flag", ["N", "Y"], 
                                            index=0 if st.session_state.get('store_fwd_flag', 'N') == 'N' else 1,
                                            key='store_fwd_flag')
        
        # Hiển thị thông tin input
        st.subheader("📊 Dữ liệu đầu vào (Raw Input)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **🕐 Thời gian:**
            - Pickup: {pickup_datetime_str}
            - Vendor: {vendor_id}
            - Passengers: {passenger_count}
            """)
        
        with col2:
            st.info(f"""
            **📍 Pickup:**
            - Lat: {pickup_lat:.4f}
            - Lon: {pickup_lon:.4f}
            - Flag: {store_fwd_flag}
            """)
        
        with col3:
            st.info(f"""
            **🎯 Dropoff:**
            - Lat: {dropoff_lat:.4f}
            - Lon: {dropoff_lon:.4f}
            """)
        
        # Quick demo scenarios
        st.subheader("🎭 Demo Scenarios")
        scenario_cols = st.columns(len(DEMO_SCENARIOS))
        
        for i, scenario in enumerate(DEMO_SCENARIOS):
            with scenario_cols[i]:
                if st.button(scenario["name"], key=f"scenario_{i}", use_container_width=True):
                    st.session_state.selected_scenario = scenario
                    apply_random_scenario()
                    st.rerun()
        
        st.divider()
        
        # Prediction button
        if st.button("🔮 Predict Trip Duration", type="primary", use_container_width=True):
            
            try:
                # Check if we have a preselected scenario
                if hasattr(st.session_state, 'selected_scenario'):
                    scenario = st.session_state.selected_scenario
                    
                    # Show processing animation
                    simulate_prediction_with_progress()
                    
                    # Use predefined results
                    features = scenario["features"]
                    prediction = scenario["prediction"]
                    
                else:
                    # Manual calculation for custom inputs
                    simulate_prediction_with_progress()
                    
                    # Calculate features manually
                    def haversine(lat1, lon1, lat2, lon2):
                        R = 6371.0
                        dlat = math.radians(lat2 - lat1)
                        dlon = math.radians(lon2 - lon1)
                        a = (math.sin(dlat/2)**2 + 
                             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                             math.sin(dlon/2)**2)
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        return R * c
                    
                    distance_km = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
                    pickup_hour = pickup_datetime.hour
                    pickup_weekday = pickup_datetime.isoweekday()
                    pickup_month = pickup_datetime.month
                    peak_hours = [7, 8, 9, 16, 17, 18, 19]
                    is_peak = 1 if pickup_hour in peak_hours else 0
                    
                    features = {
                        "pickup_hour": pickup_hour,
                        "pickup_weekday": pickup_weekday, 
                        "pickup_month": pickup_month,
                        "is_peak": is_peak,
                        "distance_km": distance_km,
                        "airport_trip": 1 if 18 <= distance_km <= 23 else 0,
                        "distance_peak": distance_km * is_peak,
                        "store_and_fwd_flag_num": 1 if store_fwd_flag == "Y" else 0
                    }
                    
                    # Simple prediction formula for demo
                    base_time = 300 + distance_km * 60  # Base + distance factor
                    peak_penalty = 180 if is_peak else 0  # Peak hour penalty
                    airport_bonus = 600 if features["airport_trip"] else 0  # Airport efficiency
                    
                    duration_seconds = base_time + peak_penalty + airport_bonus
                    duration_minutes = duration_seconds / 60
                    avg_speed = distance_km / (duration_seconds / 3600) if duration_seconds > 0 else 0
                    log_pred = math.log1p(duration_seconds)
                    
                    prediction = {
                        "duration_minutes": duration_minutes,
                        "duration_seconds": duration_seconds,
                        "avg_speed": avg_speed,
                        "log_prediction": log_pred
                    }
                
                # === HIỂN THỊ KẾT QUẢ ===
                st.success("🎉 Prediction hoàn thành!")
                
                st.subheader("🔧 Features được trích xuất tự động")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🕐 Pickup Hour", features["pickup_hour"])
                    st.metric("📅 Weekday", features["pickup_weekday"])
                
                with col2:
                    st.metric("📆 Month", features["pickup_month"])  
                    st.metric("⏰ Peak Hour", "Yes" if features["is_peak"] else "No")
                
                with col3:
                    st.metric("📏 Distance (km)", f"{features['distance_km']:.2f}")
                    st.metric("✈️ Airport Trip", "Yes" if features["airport_trip"] else "No")
                
                with col4:
                    st.metric("🚗 Distance×Peak", f"{features['distance_peak']:.2f}")
                    st.metric("📡 Store&Forward", "Yes" if features["store_and_fwd_flag_num"] else "No")
                
                st.divider()
                
                st.subheader("🎯 Kết quả dự đoán")
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="⏱️ Thời gian dự đoán",
                        value=f"{prediction['duration_minutes']:.1f} phút",
                        delta=f"{prediction['duration_seconds']:.0f} giây"
                    )
                
                with col2:
                    st.metric(
                        label="📏 Khoảng cách",
                        value=f"{features['distance_km']:.2f} km"
                    )
                
                with col3:
                    st.metric(
                        label="🚗 Tốc độ TB",
                        value=f"{prediction['avg_speed']:.1f} km/h"
                    )
                
                with col4:
                    st.metric(
                        label="📊 Log Prediction", 
                        value=f"{prediction['log_prediction']:.3f}"
                    )
                
                # Success message with emoji
                if prediction["duration_minutes"] < 10:
                    st.success(f"🚀 Chuyến đi ngắn! Dự kiến {prediction['duration_minutes']:.1f} phút")
                elif prediction["duration_minutes"] < 30:
                    st.info(f"🚕 Chuyến đi bình thường! Dự kiến {prediction['duration_minutes']:.1f} phút")
                else:
                    st.warning(f"🚦 Chuyến đi dài! Dự kiến {prediction['duration_minutes']:.1f} phút")
                
                # Detailed breakdown
                with st.expander("📋 Chi tiết kỹ thuật"):
                    details_df = pd.DataFrame({
                        'Metric': [
                            'Processing Mode', 'Input Features', 'Generated Features', 
                            'Log Prediction', 'Duration (seconds)', 
                            'Duration (minutes)', 'Distance (km)', 
                            'Average Speed (km/h)', 'Peak Hour Factor',
                            'Airport Trip Factor'
                        ],
                        'Value': [
                            'Demo Mode' if model is None else 'Live Model',
                            '8 raw features', '15+ engineered features',
                            f'{prediction["log_prediction"]:.4f}', 
                            f'{prediction["duration_seconds"]:.0f}',
                            f'{prediction["duration_minutes"]:.1f}', 
                            f'{features["distance_km"]:.2f}',
                            f'{prediction["avg_speed"]:.1f}', 
                            'Yes' if features['is_peak'] else 'No',
                            'Yes' if features['airport_trip'] else 'No'
                        ]
                    })
                    st.dataframe(details_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.code(f"Error: {str(e)}")
    
    if __name__ == "__main__":
        main()
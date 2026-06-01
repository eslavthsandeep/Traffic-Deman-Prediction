# """
# =============================================================================
# Feature Extraction Script - Traffic Demand Prediction (Flipkart GRID)
# =============================================================================
# Reads raw datasets from 'dataset/' and performs full feature engineering.
# Saves the engineered datasets to 'dataset/train_features.csv' and 'dataset/test_features.csv'.
# =============================================================================
# """

# import pandas as pd
# import numpy as np
# import os
# import warnings
# from sklearn.preprocessing import LabelEncoder

# warnings.filterwarnings('ignore')

# # Configuration
# DATA_DIR = "dataset"
# os.makedirs(DATA_DIR, exist_ok=True)

# def parse_timestamp(ts):
#     """Parse timestamp string like '0:0', '2:15', '23:45' into hour, minute"""
#     try:
#         parts = str(ts).split(':')
#         hour = int(parts[0])
#         minute = int(parts[1]) if len(parts) > 1 else 0
#         return hour, minute
#     except:
#         return np.nan, np.nan

# def main():
#     print("=" * 70)
#     print("STEP 1: LOADING RAW DATA")
#     print("=" * 70)
    
#     train_path = os.path.join(DATA_DIR, "train.csv")
#     test_path = os.path.join(DATA_DIR, "test.csv")
    
#     if not os.path.exists(train_path) or not os.path.exists(test_path):
#         raise FileNotFoundError(f"Ensure train.csv and test.csv are in '{DATA_DIR}' folder.")
        
#     train = pd.read_csv(train_path)
#     test = pd.read_csv(test_path)
    
#     print(f"Train shape: {train.shape}")
#     print(f"Test shape:  {test.shape}")
    
#     # Combine train and test for consistent preprocessing
#     train['is_train'] = 1
#     test['is_train'] = 0
#     if 'demand' not in test.columns:
#         test['demand'] = np.nan
        
#     df = pd.concat([train, test], axis=0, ignore_index=True)
#     print(f"Combined shape: {df.shape}")
    
#     print("\n" + "=" * 70)
#     print("STEP 2: PREPROCESSING & MISSING VALUE HANDLING")
#     print("=" * 70)
    
#     # 2.1 Parse timestamp -> hour and minute
#     df[['hour', 'minute']] = df['timestamp'].apply(lambda x: pd.Series(parse_timestamp(x)))
#     print(f"  Parsed timestamp -> hour ({df['hour'].min()}-{df['hour'].max()}), minute ({df['minute'].min()}-{df['minute'].max()})")
    
#     # 2.2 Handle missing RoadType
#     roadtype_mode = df['RoadType'].mode()[0]
#     df['RoadType'].fillna(roadtype_mode, inplace=True)
#     print(f"  Filled missing RoadType with mode: {roadtype_mode}")
    
#     # 2.3 Handle missing Weather
#     weather_mode = df['Weather'].mode()[0]
#     df['Weather'].fillna(weather_mode, inplace=True)
#     print(f"  Filled missing Weather with mode: {weather_mode}")
    
#     # 2.4 Handle missing Temperature (hierarchical imputation)
#     temp_fill = df.groupby(['geohash', 'hour', 'Weather'])['Temperature'].transform('median')
#     df['Temperature'] = df['Temperature'].fillna(temp_fill)
    
#     temp_fill2 = df.groupby(['hour', 'Weather'])['Temperature'].transform('median')
#     df['Temperature'] = df['Temperature'].fillna(temp_fill2)
    
#     temp_fill3 = df.groupby(['Weather'])['Temperature'].transform('median')
#     df['Temperature'] = df['Temperature'].fillna(temp_fill3)
    
#     df['Temperature'].fillna(df['Temperature'].median(), inplace=True)
#     print(f"  Filled Temperature missing values. Remaining missing: {df['Temperature'].isnull().sum()}")
    
#     print("\n" + "=" * 70)
#     print("STEP 3: FEATURE ENGINEERING")
#     print("=" * 70)
    
#     # ------------- 3.1 TIME FEATURES -------------
#     print("  [3.1] Time Features...")
#     df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
#     df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
#     def get_time_slot(hour):
#         if 0 <= hour <= 5: return 0  # Night
#         elif 6 <= hour <= 10: return 1  # Morning
#         elif 11 <= hour <= 15: return 2  # Noon
#         elif 16 <= hour <= 20: return 3  # Evening
#         else: return 4  # Late Night
        
#     df['time_slot'] = df['hour'].apply(get_time_slot)
#     df['is_morning'] = (df['hour'].between(6, 10)).astype(int)
#     df['is_afternoon'] = (df['hour'].between(11, 15)).astype(int)
#     df['is_evening'] = (df['hour'].between(16, 20)).astype(int)
#     df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 5)).astype(int)
#     df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
#     df['quarter_of_day'] = df['hour'] // 6
    
#     # ------------- 3.2 DAY FEATURES -------------
#     print("  [3.2] Day Features...")
#     df['day_of_week'] = df['day'] % 7
#     df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
#     df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
#     df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
#     df['day_normalized'] = (df['day'] - df['day'].min()) / (df['day'].max() - df['day'].min() + 1)
    
#     # ------------- 3.3 GEOHASH FEATURES -------------
#     print("  [3.3] Geohash Features...")
#     geo_freq = df['geohash'].value_counts().to_dict()
#     df['geo_frequency'] = df['geohash'].map(geo_freq)
    
#     le_geo = LabelEncoder()
#     df['geohash_encoded'] = le_geo.fit_transform(df['geohash'])
    
#     df['geohash_prefix_4'] = df['geohash'].str[:4]
#     df['geohash_prefix_5'] = df['geohash'].str[:5]
    
#     df['geohash_prefix_4_enc'] = LabelEncoder().fit_transform(df['geohash_prefix_4'])
#     df['geohash_prefix_5_enc'] = LabelEncoder().fit_transform(df['geohash_prefix_5'])
    
#     df['geo_hour'] = df['geohash'].astype(str) + '_' + df['hour'].astype(str)
#     df['geo_hour_encoded'] = LabelEncoder().fit_transform(df['geo_hour'])
    
#     df['geo_timeslot'] = df['geohash'].astype(str) + '_' + df['time_slot'].astype(str)
#     df['geo_timeslot_encoded'] = LabelEncoder().fit_transform(df['geo_timeslot'])
    
#     # Target encoding for geohash (using only training data to avoid leakage)
#     train_mask = df['is_train'] == 1
#     geo_demand_stats = df[train_mask].groupby('geohash')['demand'].agg(['mean', 'median', 'std', 'count'])
#     geo_demand_stats.columns = ['geo_demand_mean', 'geo_demand_median', 'geo_demand_std', 'geo_demand_count']
#     geo_demand_stats['geo_demand_std'] = geo_demand_stats['geo_demand_std'].fillna(0)
#     df = df.merge(geo_demand_stats, on='geohash', how='left')
    
#     global_mean = df[train_mask]['demand'].mean()
#     global_median = df[train_mask]['demand'].median()
#     global_std = df[train_mask]['demand'].std()
    
#     df['geo_demand_mean'].fillna(global_mean, inplace=True)
#     df['geo_demand_median'].fillna(global_median, inplace=True)
#     df['geo_demand_std'].fillna(global_std, inplace=True)
#     df['geo_demand_count'].fillna(0, inplace=True)
    
#     # Target encoding for geohash + hour
#     geo_hour_demand = df[train_mask].groupby(['geohash', 'hour'])['demand'].agg(['mean', 'median']).reset_index()
#     geo_hour_demand.columns = ['geohash', 'hour', 'geo_hour_demand_mean', 'geo_hour_demand_median']
#     df = df.merge(geo_hour_demand, on=['geohash', 'hour'], how='left')
#     df['geo_hour_demand_mean'].fillna(df['geo_demand_mean'], inplace=True)
#     df['geo_hour_demand_median'].fillna(df['geo_demand_median'], inplace=True)
    
#     # Target encoding for geohash + day_of_week
#     geo_dow_demand = df[train_mask].groupby(['geohash', 'day_of_week'])['demand'].agg(['mean']).reset_index()
#     geo_dow_demand.columns = ['geohash', 'day_of_week', 'geo_dow_demand_mean']
#     df = df.merge(geo_dow_demand, on=['geohash', 'day_of_week'], how='left')
#     df['geo_dow_demand_mean'].fillna(df['geo_demand_mean'], inplace=True)
    
#     # ------------- 3.4 ROAD FEATURES -------------
#     print("  [3.4] Road Features...")
#     df['RoadType_encoded'] = LabelEncoder().fit_transform(df['RoadType'])
#     df['LargeVehicles_binary'] = (df['LargeVehicles'] == 'Allowed').astype(int)
#     df['Landmarks_binary'] = (df['Landmarks'] == 'Yes').astype(int)
#     df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_binary'])
#     df['landmark_lane_interaction'] = df['Landmarks_binary'] * df['NumberofLanes']
#     df['road_complexity'] = df['NumberofLanes'] + df['LargeVehicles_binary'] + df['Landmarks_binary']
    
#     df['road_lanes'] = df['RoadType'].astype(str) + '_' + df['NumberofLanes'].astype(str)
#     df['road_lanes_encoded'] = LabelEncoder().fit_transform(df['road_lanes'])
    
#     # RoadType target encoding
#     road_demand = df[train_mask].groupby('RoadType')['demand'].agg(['mean', 'std']).reset_index()
#     road_demand.columns = ['RoadType', 'road_demand_mean', 'road_demand_std']
#     df = df.merge(road_demand, on='RoadType', how='left')
#     df['road_demand_mean'].fillna(global_mean, inplace=True)
#     df['road_demand_std'].fillna(global_std, inplace=True)
    
#     # ------------- 3.5 TEMPERATURE FEATURES -------------
#     print("  [3.5] Temperature Features...")
#     df['temp_bin'] = pd.cut(df['Temperature'], bins=[-np.inf, 10, 20, 30, np.inf],
#                             labels=[0, 1, 2, 3]).astype(float)
                            
#     def get_temp_cat(t):
#         if t < 10: return 0
#         elif t < 20: return 1
#         elif t < 30: return 2
#         else: return 3
        
#     df['temp_category'] = df['Temperature'].apply(get_temp_cat)
#     df['temp_squared'] = df['Temperature'] ** 2
#     df['temp_normalized'] = (df['Temperature'] - df['Temperature'].mean()) / (df['Temperature'].std() + 1e-8)
    
#     # ------------- 3.6 WEATHER FEATURES -------------
#     print("  [3.6] Weather Features...")
#     weather_dummies = pd.get_dummies(df['Weather'], prefix='weather')
#     df = pd.concat([df, weather_dummies], axis=1)
    
#     df['Weather_encoded'] = LabelEncoder().fit_transform(df['Weather'])
#     weather_severity = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}
#     df['weather_severity'] = df['Weather'].map(weather_severity).fillna(1)
#     df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    
#     weather_demand = df[train_mask].groupby('Weather')['demand'].agg(['mean', 'std']).reset_index()
#     weather_demand.columns = ['Weather', 'weather_demand_mean', 'weather_demand_std']
#     df = df.merge(weather_demand, on='Weather', how='left')
#     df['weather_demand_mean'].fillna(global_mean, inplace=True)
#     df['weather_demand_std'].fillna(global_std, inplace=True)
    
#     # ------------- 3.7 INTERACTION FEATURES -------------
#     print("  [3.7] Interaction Features...")
#     df['geo_road'] = df['geohash'].astype(str) + '_' + df['RoadType'].astype(str)
#     df['geo_road_encoded'] = LabelEncoder().fit_transform(df['geo_road'])
    
#     df['geo_weather'] = df['geohash'].astype(str) + '_' + df['Weather'].astype(str)
#     df['geo_weather_encoded'] = LabelEncoder().fit_transform(df['geo_weather'])
    
#     df['road_weather'] = df['RoadType'].astype(str) + '_' + df['Weather'].astype(str)
#     df['road_weather_encoded'] = LabelEncoder().fit_transform(df['road_weather'])
    
#     df['road_type_lanes'] = df['RoadType_encoded'] * 10 + df['NumberofLanes']
    
#     df['hour_weather'] = df['hour'].astype(str) + '_' + df['Weather'].astype(str)
#     df['hour_weather_encoded'] = LabelEncoder().fit_transform(df['hour_weather'])
    
#     df['geo_dow'] = df['geohash'].astype(str) + '_' + df['day_of_week'].astype(str)
#     df['geo_dow_encoded'] = LabelEncoder().fit_transform(df['geo_dow'])
    
#     # Target encoding for key interactions
#     for interact_cols, name in [
#         (['geohash', 'RoadType'], 'geo_road'),
#         (['geohash', 'Weather'], 'geo_weather'),
#         (['hour', 'Weather'], 'hour_weather'),
#         (['RoadType', 'Weather'], 'road_weather'),
#     ]:
#         col_name = f'{name}_demand_mean'
#         interact_demand = df[train_mask].groupby(interact_cols)['demand'].mean().reset_index()
#         interact_demand.columns = interact_cols + [col_name]
#         df = df.merge(interact_demand, on=interact_cols, how='left')
#         df[col_name].fillna(global_mean, inplace=True)
        
#     # ------------- 3.8 AGGREGATE/STATISTICAL FEATURES -------------
#     print("  [3.8] Aggregate Features...")
#     hour_demand = df[train_mask].groupby('hour')['demand'].agg(['mean', 'std', 'median']).reset_index()
#     hour_demand.columns = ['hour', 'hour_demand_mean', 'hour_demand_std', 'hour_demand_median']
#     df = df.merge(hour_demand, on='hour', how='left')
#     df['hour_demand_mean'].fillna(global_mean, inplace=True)
#     df['hour_demand_std'].fillna(global_std, inplace=True)
#     df['hour_demand_median'].fillna(global_median, inplace=True)
    
#     ts_demand = df[train_mask].groupby('time_slot')['demand'].agg(['mean', 'median']).reset_index()
#     ts_demand.columns = ['time_slot', 'timeslot_demand_mean', 'timeslot_demand_median']
#     df = df.merge(ts_demand, on='time_slot', how='left')
#     df['timeslot_demand_mean'].fillna(global_mean, inplace=True)
#     df['timeslot_demand_median'].fillna(global_median, inplace=True)
    
#     dow_demand = df[train_mask].groupby('day_of_week')['demand'].agg(['mean', 'median']).reset_index()
#     dow_demand.columns = ['day_of_week', 'dow_demand_mean', 'dow_demand_median']
#     df = df.merge(dow_demand, on='day_of_week', how='left')
#     df['dow_demand_mean'].fillna(global_mean, inplace=True)
#     df['dow_demand_median'].fillna(global_median, inplace=True)
    
#     lane_demand = df[train_mask].groupby('NumberofLanes')['demand'].agg(['mean']).reset_index()
#     lane_demand.columns = ['NumberofLanes', 'lane_demand_mean']
#     df = df.merge(lane_demand, on='NumberofLanes', how='left')
#     df['lane_demand_mean'].fillna(global_mean, inplace=True)
    
#     # ------------- 3.9 PREPARE FOR EXPORT -------------
#     print("\n" + "=" * 70)
#     print("STEP 4: SAVING FEATURE-EXTRACTED DATASETS")
#     print("=" * 70)
    
#     # Columns to exclude from ML features (they are descriptive columns or strings)
#     exclude_cols = [
#         'timestamp', 'geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
#         'geohash_prefix_4', 'geohash_prefix_5',
#         'geo_hour', 'geo_timeslot', 'geo_road', 'geo_weather',
#         'road_weather', 'hour_weather', 'road_lanes', 'geo_dow'
#     ]
    
#     # Columns to keep
#     # We keep 'is_train', 'demand', and 'Index' for routing and matching
#     keep_cols = [c for c in df.columns if c not in exclude_cols]
    
#     # Ensure all feature columns are numeric
#     feature_cols = [c for c in keep_cols if c not in ['demand', 'is_train', 'Index']]
#     for col in feature_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
#     # Filter to only keep numeric feature columns + Index + demand
#     train_keep = [c for c in keep_cols if c != 'is_train']
#     test_keep = [c for c in keep_cols if c not in ['is_train', 'demand']]
    
#     train_features = df.loc[df['is_train'] == 1, train_keep]
#     test_features = df.loc[df['is_train'] == 0, test_keep]
    
#     # Assert check
#     print(f"Final Train Features shape: {train_features.shape}")
#     print(f"Final Test Features shape:  {test_features.shape}")
    
#     train_features_path = os.path.join(DATA_DIR, "train_features.csv")
#     test_features_path = os.path.join(DATA_DIR, "test_features.csv")
    
#     train_features.to_csv(train_features_path, index=False)
#     print(f"  [SAVED] {train_features_path}")
    
#     test_features.to_csv(test_features_path, index=False)
#     print(f"  [SAVED] {test_features_path}")
    
#     print("\nFEATURE EXTRACTION PIPELINE COMPLETE!")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()


"""
=============================================================================
Feature Extraction Script - Traffic Demand Prediction (Flipkart GRID)
=============================================================================
Reads raw datasets from 'dataset/' and performs full feature engineering.
Saves the engineered datasets to 'dataset/train_features.csv' and 'dataset/test_features.csv'.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

def parse_timestamp(ts):
    """Parse timestamp string like '0:0', '2:15', '23:45' into hour, minute"""
    try:
        parts = str(ts).split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return hour, minute
    except:
        return np.nan, np.nan

def main():
    print("=" * 70)
    print("STEP 1: LOADING RAW DATA")
    print("=" * 70)
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Ensure train.csv and test.csv are in '{DATA_DIR}' folder.")
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    
    # Combine train and test for consistent preprocessing
    train['is_train'] = 1
    test['is_train'] = 0
    if 'demand' not in test.columns:
        test['demand'] = np.nan
        
    df = pd.concat([train, test], axis=0, ignore_index=True)
    print(f"Combined shape: {df.shape}")
    
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING & MISSING VALUE HANDLING")
    print("=" * 70)
    
    # 2.1 Parse timestamp -> hour and minute
    df[['hour', 'minute']] = df['timestamp'].apply(lambda x: pd.Series(parse_timestamp(x)))
    print(f"  Parsed timestamp -> hour ({df['hour'].min()}-{df['hour'].max()}), minute ({df['minute'].min()}-{df['minute'].max()})")
    
    # 2.2 Handle missing RoadType
    roadtype_mode = df['RoadType'].mode()[0]
    df['RoadType'].fillna(roadtype_mode, inplace=True)
    print(f"  Filled missing RoadType with mode: {roadtype_mode}")
    
    # 2.3 Handle missing Weather
    weather_mode = df['Weather'].mode()[0]
    df['Weather'].fillna(weather_mode, inplace=True)
    print(f"  Filled missing Weather with mode: {weather_mode}")
    
    # 2.4 Handle missing Temperature (hierarchical imputation)
    temp_fill = df.groupby(['geohash', 'hour', 'Weather'])['Temperature'].transform('median')
    df['Temperature'] = df['Temperature'].fillna(temp_fill)
    
    temp_fill2 = df.groupby(['hour', 'Weather'])['Temperature'].transform('median')
    df['Temperature'] = df['Temperature'].fillna(temp_fill2)
    
    temp_fill3 = df.groupby(['Weather'])['Temperature'].transform('median')
    df['Temperature'] = df['Temperature'].fillna(temp_fill3)
    
    df['Temperature'].fillna(df['Temperature'].median(), inplace=True)
    print(f"  Filled Temperature missing values. Remaining missing: {df['Temperature'].isnull().sum()}")
    
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    # ------------- 3.1 TIME FEATURES -------------
    print("  [3.1] Time Features...")
    df['minutes_since_midnight'] = df['hour'] * 60 + df['minute']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    def get_time_slot(hour):
        if 0 <= hour <= 5: return 0  # Night
        elif 6 <= hour <= 10: return 1  # Morning
        elif 11 <= hour <= 15: return 2  # Noon
        elif 16 <= hour <= 20: return 3  # Evening
        else: return 4  # Late Night
        
    df['time_slot'] = df['hour'].apply(get_time_slot)
    df['is_morning'] = (df['hour'].between(6, 10)).astype(int)
    df['is_afternoon'] = (df['hour'].between(11, 15)).astype(int)
    df['is_evening'] = (df['hour'].between(16, 20)).astype(int)
    df['is_night'] = ((df['hour'] >= 21) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
    df['quarter_of_day'] = df['hour'] // 6
    
    # ------------- 3.2 DAY FEATURES -------------
    print("  [3.2] Day Features...")
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_normalized'] = (df['day'] - df['day'].min()) / (df['day'].max() - df['day'].min() + 1)
    
    # ------------- 3.3 GEOHASH FEATURES -------------
    print("  [3.3] Geohash Features...")
    geo_freq = df['geohash'].value_counts().to_dict()
    df['geo_frequency'] = df['geohash'].map(geo_freq)
    
    le_geo = LabelEncoder()
    df['geohash_encoded'] = le_geo.fit_transform(df['geohash'])
    
    df['geohash_prefix_4'] = df['geohash'].str[:4]
    df['geohash_prefix_5'] = df['geohash'].str[:5]
    
    df['geohash_prefix_4_enc'] = LabelEncoder().fit_transform(df['geohash_prefix_4'])
    df['geohash_prefix_5_enc'] = LabelEncoder().fit_transform(df['geohash_prefix_5'])
    
    df['geo_hour'] = df['geohash'].astype(str) + '_' + df['hour'].astype(str)
    df['geo_hour_encoded'] = LabelEncoder().fit_transform(df['geo_hour'])
    
    df['geo_timeslot'] = df['geohash'].astype(str) + '_' + df['time_slot'].astype(str)
    df['geo_timeslot_encoded'] = LabelEncoder().fit_transform(df['geo_timeslot'])
    
    # Target encoding for geohash (using only training data to avoid leakage)
    train_mask = df['is_train'] == 1
    geo_demand_stats = df[train_mask].groupby('geohash')['demand'].agg(['mean', 'median', 'std', 'count'])
    geo_demand_stats.columns = ['geo_demand_mean', 'geo_demand_median', 'geo_demand_std', 'geo_demand_count']
    geo_demand_stats['geo_demand_std'] = geo_demand_stats['geo_demand_std'].fillna(0)
    df = df.merge(geo_demand_stats, on='geohash', how='left')
    
    global_mean = df[train_mask]['demand'].mean()
    global_median = df[train_mask]['demand'].median()
    global_std = df[train_mask]['demand'].std()
    
    df['geo_demand_mean'].fillna(global_mean, inplace=True)
    df['geo_demand_median'].fillna(global_median, inplace=True)
    df['geo_demand_std'].fillna(global_std, inplace=True)
    df['geo_demand_count'].fillna(0, inplace=True)
    
    # Target encoding for geohash + hour
    geo_hour_demand = df[train_mask].groupby(['geohash', 'hour'])['demand'].agg(['mean', 'median']).reset_index()
    geo_hour_demand.columns = ['geohash', 'hour', 'geo_hour_demand_mean', 'geo_hour_demand_median']
    df = df.merge(geo_hour_demand, on=['geohash', 'hour'], how='left')
    df['geo_hour_demand_mean'].fillna(df['geo_demand_mean'], inplace=True)
    df['geo_hour_demand_median'].fillna(df['geo_demand_median'], inplace=True)
    
    # Target encoding for geohash + day_of_week
    geo_dow_demand = df[train_mask].groupby(['geohash', 'day_of_week'])['demand'].agg(['mean']).reset_index()
    geo_dow_demand.columns = ['geohash', 'day_of_week', 'geo_dow_demand_mean']
    df = df.merge(geo_dow_demand, on=['geohash', 'day_of_week'], how='left')
    df['geo_dow_demand_mean'].fillna(df['geo_demand_mean'], inplace=True)
    
    # ------------- 3.4 ROAD FEATURES -------------
    print("  [3.4] Road Features...")
    df['RoadType_encoded'] = LabelEncoder().fit_transform(df['RoadType'])
    df['LargeVehicles_binary'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_binary'] = (df['Landmarks'] == 'Yes').astype(int)
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_binary'])
    df['landmark_lane_interaction'] = df['Landmarks_binary'] * df['NumberofLanes']
    df['road_complexity'] = df['NumberofLanes'] + df['LargeVehicles_binary'] + df['Landmarks_binary']
    
    df['road_lanes'] = df['RoadType'].astype(str) + '_' + df['NumberofLanes'].astype(str)
    df['road_lanes_encoded'] = LabelEncoder().fit_transform(df['road_lanes'])
    
    # RoadType target encoding
    road_demand = df[train_mask].groupby('RoadType')['demand'].agg(['mean', 'std']).reset_index()
    road_demand.columns = ['RoadType', 'road_demand_mean', 'road_demand_std']
    df = df.merge(road_demand, on='RoadType', how='left')
    df['road_demand_mean'].fillna(global_mean, inplace=True)
    df['road_demand_std'].fillna(global_std, inplace=True)
    
    # ------------- 3.5 TEMPERATURE FEATURES -------------
    print("  [3.5] Temperature Features...")
    df['temp_bin'] = pd.cut(df['Temperature'], bins=[-np.inf, 10, 20, 30, np.inf],
                            labels=[0, 1, 2, 3]).astype(float)
                            
    def get_temp_cat(t):
        if t < 10: return 0
        elif t < 20: return 1
        elif t < 30: return 2
        else: return 3
        
    df['temp_category'] = df['Temperature'].apply(get_temp_cat)
    df['temp_squared'] = df['Temperature'] ** 2
    df['temp_normalized'] = (df['Temperature'] - df['Temperature'].mean()) / (df['Temperature'].std() + 1e-8)
    
    # ------------- 3.6 WEATHER FEATURES -------------
    print("  [3.6] Weather Features...")
    weather_dummies = pd.get_dummies(df['Weather'], prefix='weather')
    df = pd.concat([df, weather_dummies], axis=1)
    
    df['Weather_encoded'] = LabelEncoder().fit_transform(df['Weather'])
    weather_severity = {'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}
    df['weather_severity'] = df['Weather'].map(weather_severity).fillna(1)
    df['is_bad_weather'] = df['Weather'].isin(['Rainy', 'Snowy', 'Foggy']).astype(int)
    
    weather_demand = df[train_mask].groupby('Weather')['demand'].agg(['mean', 'std']).reset_index()
    weather_demand.columns = ['Weather', 'weather_demand_mean', 'weather_demand_std']
    df = df.merge(weather_demand, on='Weather', how='left')
    df['weather_demand_mean'].fillna(global_mean, inplace=True)
    df['weather_demand_std'].fillna(global_std, inplace=True)
    
    # ------------- 3.7 INTERACTION FEATURES -------------
    print("  [3.7] Interaction Features...")
    df['geo_road'] = df['geohash'].astype(str) + '_' + df['RoadType'].astype(str)
    df['geo_road_encoded'] = LabelEncoder().fit_transform(df['geo_road'])
    
    df['geo_weather'] = df['geohash'].astype(str) + '_' + df['Weather'].astype(str)
    df['geo_weather_encoded'] = LabelEncoder().fit_transform(df['geo_weather'])
    
    df['road_weather'] = df['RoadType'].astype(str) + '_' + df['Weather'].astype(str)
    df['road_weather_encoded'] = LabelEncoder().fit_transform(df['road_weather'])
    
    df['road_type_lanes'] = df['RoadType_encoded'] * 10 + df['NumberofLanes']
    
    df['hour_weather'] = df['hour'].astype(str) + '_' + df['Weather'].astype(str)
    df['hour_weather_encoded'] = LabelEncoder().fit_transform(df['hour_weather'])
    
    df['geo_dow'] = df['geohash'].astype(str) + '_' + df['day_of_week'].astype(str)
    df['geo_dow_encoded'] = LabelEncoder().fit_transform(df['geo_dow'])
    
    # Target encoding for key interactions
    for interact_cols, name in [
        (['geohash', 'RoadType'], 'geo_road'),
        (['geohash', 'Weather'], 'geo_weather'),
        (['hour', 'Weather'], 'hour_weather'),
        (['RoadType', 'Weather'], 'road_weather'),
    ]:
        col_name = f'{name}_demand_mean'
        interact_demand = df[train_mask].groupby(interact_cols)['demand'].mean().reset_index()
        interact_demand.columns = interact_cols + [col_name]
        df = df.merge(interact_demand, on=interact_cols, how='left')
        df[col_name].fillna(global_mean, inplace=True)
        
    # ------------- 3.8 AGGREGATE/STATISTICAL FEATURES -------------
    print("  [3.8] Aggregate Features...")
    hour_demand = df[train_mask].groupby('hour')['demand'].agg(['mean', 'std', 'median']).reset_index()
    hour_demand.columns = ['hour', 'hour_demand_mean', 'hour_demand_std', 'hour_demand_median']
    df = df.merge(hour_demand, on='hour', how='left')
    df['hour_demand_mean'].fillna(global_mean, inplace=True)
    df['hour_demand_std'].fillna(global_std, inplace=True)
    df['hour_demand_median'].fillna(global_median, inplace=True)
    
    ts_demand = df[train_mask].groupby('time_slot')['demand'].agg(['mean', 'median']).reset_index()
    ts_demand.columns = ['time_slot', 'timeslot_demand_mean', 'timeslot_demand_median']
    df = df.merge(ts_demand, on='time_slot', how='left')
    df['timeslot_demand_mean'].fillna(global_mean, inplace=True)
    df['timeslot_demand_median'].fillna(global_median, inplace=True)
    
    dow_demand = df[train_mask].groupby('day_of_week')['demand'].agg(['mean', 'median']).reset_index()
    dow_demand.columns = ['day_of_week', 'dow_demand_mean', 'dow_demand_median']
    df = df.merge(dow_demand, on='day_of_week', how='left')
    df['dow_demand_mean'].fillna(global_mean, inplace=True)
    df['dow_demand_median'].fillna(global_median, inplace=True)
    
    lane_demand = df[train_mask].groupby('NumberofLanes')['demand'].agg(['mean']).reset_index()
    lane_demand.columns = ['NumberofLanes', 'lane_demand_mean']
    df = df.merge(lane_demand, on='NumberofLanes', how='left')
    df['lane_demand_mean'].fillna(global_mean, inplace=True)
    
    # ------------- 3.9 PREPARE FOR EXPORT -------------
    print("\n" + "=" * 70)
    print("STEP 4: SAVING FEATURE-EXTRACTED DATASETS")
    print("=" * 70)
    
    # Columns to exclude from ML features (they are descriptive columns or strings)
    exclude_cols = [
        'timestamp', 'geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
        'geohash_prefix_4', 'geohash_prefix_5',
        'geo_hour', 'geo_timeslot', 'geo_road', 'geo_weather',
        'road_weather', 'hour_weather', 'road_lanes', 'geo_dow'
    ]
    
    # Columns to keep
    # We keep 'is_train', 'demand', and 'Index' for routing and matching
    keep_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Ensure all feature columns are numeric
    feature_cols = [c for c in keep_cols if c not in ['demand', 'is_train', 'Index']]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
    # Filter to only keep numeric feature columns + Index + demand
    train_keep = [c for c in keep_cols if c != 'is_train']
    test_keep = [c for c in keep_cols if c not in ['is_train', 'demand']]
    
    train_features = df.loc[df['is_train'] == 1, train_keep]
    test_features = df.loc[df['is_train'] == 0, test_keep]
    
    # Assert check
    print(f"Final Train Features shape: {train_features.shape}")
    print(f"Final Test Features shape:  {test_features.shape}")
    
    train_features_path = os.path.join(DATA_DIR, "train_features.csv")
    test_features_path = os.path.join(DATA_DIR, "test_features.csv")
    
    train_features.to_csv(train_features_path, index=False)
    print(f"  [SAVED] {train_features_path}")
    
    test_features.to_csv(test_features_path, index=False)
    print(f"  [SAVED] {test_features_path}")
    
    print("\nFEATURE EXTRACTION PIPELINE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()

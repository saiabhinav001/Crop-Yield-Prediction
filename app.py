# --- 1. Import All Necessary Libraries ---
# Core libraries for building the web service and handling data
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import time
from contextlib import asynccontextmanager

# Core libraries for the machine learning pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import r2_score

# --- 2. Global Variables: Application State ---
# These variables will hold the fully trained model and all necessary helper objects.
# They are initialized as None and will be populated once, during the application's startup.
MODEL_PACKAGE = None
ENCODERS = {}
MIN_YEAR = None
FEATURE_ORDER = None
SOIL_DF = None
IRRIGATION_DF = None
WEATHER_LOOKUP = None
WEATHER_FALLBACK_LOOKUP = None

# --- 3. The Core Training Pipeline Function ---
def train_model_pipeline():
    """
    This is the main engine of the application. It replicates the entire model
    development process from your Jupyter Notebook, running from scratch on startup.
    It handles data loading, cleaning, feature engineering, and state-of-the-art
    ensemble model training.
    """
    print("--- [STARTUP] KICKING OFF MODEL TRAINING PIPELINE ---")
    
    # --- Step 1: Load Raw Datasets ---
    try:
        apy_df = pd.read_csv('odisha_data/merged_apy_data.csv')
        soil_df = pd.read_csv('odisha_data/odisha_soil_data.csv')
        irrigation_df = pd.read_csv('odisha_data/odisha_irrigation_data.csv')
        print("[INFO] Step 1: All data files loaded successfully.")
    except FileNotFoundError as e:
        raise Exception(f"CRITICAL ERROR: Could not find a required data file: {e}")

    # --- Step 2: Merge DataFrames into a Single Master Table ---
    # We merge first to ensure all subsequent operations are on one unified dataframe.
    print("[INFO] Step 2: Merging dataframes...")
    merged_df = pd.merge(apy_df, soil_df, left_on='District_Name_x', right_on='District', how='left')
    merged_df = pd.merge(merged_df, irrigation_df, left_on='District_Name_x', right_on='District', how='left')
    print("[SUCCESS] Step 2: Dataframes merged successfully.")

    # --- Step 3: Bulletproof Data Cleaning and Standardization ---
    print("[INFO] Step 3: Starting robust data cleaning and standardization...")
    
    # A. Standardize all column names to a consistent uppercase format.
    merged_df.columns = [col.strip().upper() for col in merged_df.columns]
    
    # B. Explicitly define which columns contain text data that needs cleaning.
    # THIS IS THE PERMANENT FIX: We no longer guess the column type; we declare it.
    # This prevents the 'upper' error on numeric columns.
    text_columns_to_clean = [
        'STATE', 'DISTRICT_NAME_X', 'CROP', 'SEASON_X', 
        'SEASON_Y', 'DISTRICT_NAME_Y', 'DISTRICT_X', 'DISTRICT_Y'
    ]
    
    # C. Precisely clean the content of the specified text-based columns.
    for col in text_columns_to_clean:
        # Check if the column exists before trying to clean it.
        if col in merged_df.columns:
            # Fill any potential missing values with an empty string before applying string functions.
            merged_df[col] = merged_df[col].fillna('').astype(str).str.strip().str.upper()

    print("[SUCCESS] Step 3: Data cleaning and standardization complete.")

    # --- Step 4: Feature Engineering & Final Preprocessing ---
    print("[INFO] Step 4: Performing feature engineering and final preprocessing...")
    
    # A. Clean up the merged dataframe by dropping redundant columns and renaming for clarity.
    merged_df = merged_df.drop(columns=['DISTRICT_NAME_Y', 'SEASON_Y', 'DISTRICT_X', 'DISTRICT_Y'], errors='ignore')
    merged_df = merged_df.rename(columns={'DISTRICT_NAME_X': 'DISTRICT', 'SEASON_X': 'SEASON'})

    # B. Create a time-based feature to capture trends over the years.
    local_min_year = merged_df['YEAR'].min()
    merged_df['YEARS_SINCE_START'] = merged_df['YEAR'] - local_min_year

    # C. Numerically encode categorical features for the models and store the encoders.
    categorical_cols = ['DISTRICT', 'CROP', 'SEASON']
    local_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on the cleaned, full column to learn all possible categories
        le.fit(merged_df[col])
        # Transform the column to its numeric representation
        merged_df[col] = le.transform(merged_df[col])
        local_encoders[col] = le

    # D. Define the final set of features (X) and the target variable (y).
    X = merged_df.drop(['YIELD', 'PRODUCTION', 'AREA', 'STATE'], axis=1)
    y = merged_df['YIELD']
    local_feature_order = X.columns.tolist()
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]
    print("[SUCCESS] Step 4: Data preparation complete.")

    # --- Step 5: Define the Blending Ensemble Models ---
    print("[INFO] Step 5: Defining optimized base models...")
    lgbm = lgb.LGBMRegressor(objective='regression_l1', n_estimators=800, learning_rate=0.05, num_leaves=120, max_depth=10, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=700, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, gamma=1, n_jobs=-1, seed=42)
    cat = cb.CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=10, l2_leaf_reg=3, loss_function='RMSE', cat_features=cat_features_indices, random_seed=42, verbose=0)
    meta_model = lgb.LGBMRegressor(objective='regression', n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1, seed=42)
    print("[SUCCESS] Step 5: All models defined.")

    # --- Step 6: K-Fold Cross-Validation Training ---
    print("[INFO] Step 6: Starting K-Fold training (this will take several minutes)...")
    start_time = time.time()
    NFOLDS = 5
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_preds_lgbm, oof_preds_xgb, oof_preds_cat = (np.zeros(X.shape[0]), np.zeros(X.shape[0]), np.zeros(X.shape[0]))

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, _ = X.iloc[valid_idx], y.iloc[valid_idx]
        print(f"--- Training Fold {n_fold + 1}/{NFOLDS} ---")
        lgbm.fit(X_train, y_train)
        oof_preds_lgbm[valid_idx] = lgbm.predict(X_valid)
        xgb_model.fit(X_train, y_train)
        oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid)
        cat.fit(X_train, y_train)
        oof_preds_cat[valid_idx] = cat.predict(X_valid)
    print("[SUCCESS] Step 6: Base model training complete.")

    # --- Step 7: Final Meta-Model Training ---
    print("[INFO] Step 7: Training the meta-model...")
    X_meta_train = pd.DataFrame({'lgbm_pred': oof_preds_lgbm, 'xgb_pred': oof_preds_xgb, 'cat_pred': oof_preds_cat})
    meta_model.fit(X_meta_train, y)
    end_time = time.time()
    final_r2 = r2_score(y, meta_model.predict(X_meta_train))
    print(f"--- [TRAINING COMPLETE] Final R-squared: {final_r2:.4f}, Total Time: {end_time - start_time:.2f}s ---")
    
    local_model_package = {'lgbm': lgbm, 'xgb': xgb_model, 'cat': cat, 'meta_model': meta_model}
    # Return the original, cleaned dataframes for the lookups
    return local_model_package, local_encoders, local_min_year, local_feature_order, soil_df, irrigation_df, apy_df

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] Application startup...")
    global MODEL_PACKAGE, ENCODERS, MIN_YEAR, FEATURE_ORDER, SOIL_DF, IRRIGATION_DF, WEATHER_LOOKUP, WEATHER_FALLBACK_LOOKUP
    try:
        model_package, encoders, min_year, feature_order, soil_df, irrigation_df, apy_df = train_model_pipeline()
        MODEL_PACKAGE, ENCODERS, MIN_YEAR, FEATURE_ORDER = model_package, encoders, min_year, feature_order
        # Build lookups from the cleaned, original dataframes
        SOIL_DF = soil_df.set_index('DISTRICT')
        IRRIGATION_DF = irrigation_df.set_index('DISTRICT')
        WEATHER_LOOKUP = apy_df.groupby(['DISTRICT_NAME_X', 'SEASON_X', 'CROP'])[['AVG_TEMP', 'MAX_TEMP', 'MIN_TEMP', 'TOTAL_RAINFALL']].mean()
        WEATHER_FALLBACK_LOOKUP = apy_df.groupby(['DISTRICT_NAME_X', 'SEASON_X'])[['AVG_TEMP', 'MAX_TEMP', 'MIN_TEMP', 'TOTAL_RAINFALL']].mean()
    except Exception as e:
        print(f"FATAL ERROR during model training on startup: {e}")
        MODEL_PACKAGE = None
    yield
    print("[LIFESPAN] Application shutdown.")

# --- FastAPI Application Setup ---
app = FastAPI(title="Crop Yield Training & Prediction API", lifespan=lifespan, version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Prediction Endpoint ---
@app.get("/predict/")
def predict_yield(district: str, crop: str, season: str, year: int):
    if not MODEL_PACKAGE: raise HTTPException(status_code=503, detail="Model is not available due to a training error on startup.")
    try:
        district_std, crop_std, season_std = district.strip().upper(), crop.strip().upper(), season.strip().upper()
        soil_data, irrigation_data = SOIL_DF.loc[district_std], IRRIGATION_DF.loc[district_std]
        try: weather_data = WEATHER_LOOKUP.loc[(district_std, season_std, crop_std)]
        except KeyError: weather_data = WEATHER_FALLBACK_LOOKUP.loc[(district_std, season_std)]
        
        new_sample = {
            'YEAR': year, 'SEASON': season_std, 'CROP': crop_std, 'DISTRICT': district_std,
            'AVG_TEMP': weather_data['AVG_TEMP'], 'MAX_TEMP': weather_data['MAX_TEMP'],
            'MIN_TEMP': weather_data['MIN_TEMP'], 'TOTAL_RAINFALL': weather_data['TOTAL_RAINFALL'],
            'N': soil_data['N'], 'P': soil_data['P'], 'K': soil_data['K'], 'PH': soil_data['PH'],
            'OC': soil_data['OC'], 'NET GROUND WATER AVAILABILITY FOR FUTURE USE (HAM)': irrigation_data['NET GROUND WATER AVAILABILITY FOR FUTURE USE (HAM)']
        }
        
        sample_df = pd.DataFrame([new_sample])
        sample_df['YEARS_SINCE_START'] = sample_df['YEAR'] - MIN_YEAR
        for col in ['DISTRICT', 'CROP', 'SEASON']:
            sample_df[col] = ENCODERS[col].transform([sample_df.iloc[0][col]])[0]
        
        sample_df = sample_df[FEATURE_ORDER]

        lgbm_pred = MODEL_PACKAGE['lgbm'].predict(sample_df)
        xgb_pred = MODEL_PACKAGE['xgb'].predict(sample_df)
        cat_pred = MODEL_PACKAGE['cat'].predict(sample_df)
        meta_features = pd.DataFrame({'lgbm_pred': lgbm_pred, 'xgb_pred': xgb_pred, 'cat_pred': cat_pred})
        final_prediction = MODEL_PACKAGE['meta_model'].predict(meta_features)

        return {"predicted_yield_tonnes_per_hectare": final_prediction[0]}
    except KeyError as e: raise HTTPException(status_code=400, detail=f"Invalid input category provided: {e}. This value was not seen during training.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {e}")

# --- API Root Endpoint (for health checks) ---
@app.get("/")
def read_root():
    status = "ready" if MODEL_PACKAGE else "training_failed_on_startup"
    return {"status": status, "message": "Crop Yield Prediction API"}


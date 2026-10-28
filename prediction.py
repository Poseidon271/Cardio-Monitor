import os
import joblib
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, "heartmodel.pkl")
_SCALER_PATH = os.path.join(_BASE_DIR, "scaler.pkl")

# Load trained model once during initialization
clfr = joblib.load(_MODEL_PATH)

# Load pre-fitted scaler once during initialization, or build and cache if not present
if os.path.exists(_SCALER_PATH):
    scaler = joblib.load(_SCALER_PATH)
else:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv(os.path.join(_BASE_DIR, 'heart.csv'))
    _y = _df["target"]
    _X = _df.drop('target', axis=1)
    _X_train, _, _, _ = train_test_split(_X, _y, test_size=0.20, random_state=0)
    scaler = StandardScaler()
    scaler.fit(_X_train.values)
    joblib.dump(scaler, _SCALER_PATH)

def validate_and_convert_features(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):
    """
    Validates all 13 features for the heart disease prediction model.
    Ensures no missing, empty, or whitespace-only values.
    Converts and encodes all inputs to safe floating-point values.
    Preserves valid zero values (e.g. oldpeak=0, ca=0).
    Raises ValueError with a user-friendly message on invalid or missing data.
    """
    raw_inputs = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'restecg': restecg,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # 1. Detect missing values, empty strings, and whitespace-only strings
    for field_name, value in raw_inputs.items():
        if value is None:
            raise ValueError(f"Please provide a value for '{field_name}'. Field cannot be empty.")
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed == "" or trimmed.lower() == "none":
                raise ValueError(f"Please provide a valid value for '{field_name}'. Field cannot be empty.")

    # 2. Validate and convert numeric fields safely
    # Age
    try:
        age_val = float(age)
        if not np.isfinite(age_val) or age_val <= 0 or age_val > 120:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid age '{age}'. Age must be a positive number between 1 and 120.")

    # Resting blood pressure (trestbps)
    try:
        trestbps_val = float(trestbps)
        if not np.isfinite(trestbps_val) or trestbps_val <= 0 or trestbps_val > 300:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid resting blood pressure '{trestbps}'. Must be a positive number between 50 and 300.")

    # Serum cholesterol (chol)
    try:
        chol_val = float(chol)
        if not np.isfinite(chol_val) or chol_val <= 0 or chol_val > 800:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid cholesterol value '{chol}'. Must be a positive number between 50 and 800.")

    # Maximum heart rate achieved (thalach)
    try:
        thalach_val = float(thalach)
        if not np.isfinite(thalach_val) or thalach_val <= 0 or thalach_val > 300:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid maximum heart rate '{thalach}'. Must be a positive number between 50 and 300.")

    # ST depression (oldpeak) - 0 is a valid value
    try:
        oldpeak_val = float(oldpeak)
        if not np.isfinite(oldpeak_val) or oldpeak_val < 0 or oldpeak_val > 15:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid ST depression (oldpeak) '{oldpeak}'. Must be a non-negative number (0 or greater).")

    # Major vessels (ca) - 0 is a valid value
    try:
        ca_val = float(ca)
        if not np.isfinite(ca_val) or ca_val < 0 or ca_val > 3:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError(f"Invalid number of major vessels (ca) '{ca}'. Must be between 0 and 3.")

    # 3. Validate and encode categorical features
    # Sex
    sex_str = str(sex).strip().lower()
    if sex_str in ["male", "1", "1.0", "m"]:
        sex_val = 1.0
    elif sex_str in ["female", "0", "0.0", "f"]:
        sex_val = 0.0
    else:
        raise ValueError(f"Invalid Gender '{sex}'. Please select Male or Female.")

    # Chest pain type (cp)
    cp_str = str(cp).strip()
    cp_lower = cp_str.lower()
    if cp_lower in ["typical angina", "0", "0.0"]:
        cp_val = 0.0
    elif cp_lower in ["atypical angina", "1", "1.0"]:
        cp_val = 1.0
    elif cp_lower in ["non-anginal pain", "2", "2.0"]:
        cp_val = 2.0
    elif cp_lower in ["asymptomatic", "3", "3.0"]:
        cp_val = 3.0
    else:
        raise ValueError(f"Invalid chest pain type '{cp}'. Please select a valid chest pain option.")

    # Exercise induced angina (exang)
    exang_str = str(exang).strip().lower()
    if exang_str in ["yes", "1", "1.0"]:
        exang_val = 1.0
    elif exang_str in ["no", "0", "0.0"]:
        exang_val = 0.0
    else:
        raise ValueError(f"Invalid exercise induced angina '{exang}'. Please select Yes or No.")

    # Fasting blood sugar (fbs)
    fbs_str = str(fbs).strip().lower()
    if fbs_str in ["yes", "1", "1.0"]:
        fbs_val = 1.0
    elif fbs_str in ["no", "0", "0.0"]:
        fbs_val = 0.0
    else:
        raise ValueError(f"Invalid fasting blood sugar '{fbs}'. Please select Yes or No.")

    # Slope
    slope_str = str(slope).strip()
    slope_lower = slope_str.lower()
    if "upsloping" in slope_lower or slope_lower in ["0", "0.0"]:
        slope_val = 0.0
    elif "flatsloping" in slope_lower or slope_lower in ["1", "1.0"]:
        slope_val = 1.0
    elif "downsloping" in slope_lower or slope_lower in ["2", "2.0"]:
        slope_val = 2.0
    else:
        raise ValueError(f"Invalid slope value '{slope}'. Please select a valid slope option.")

    # Thal
    thal_str = str(thal).strip()
    thal_lower = thal_str.lower()
    if "fixed defect" in thal_lower or thal_lower in ["6", "6.0"]:
        thal_val = 6.0
    elif "reversable defect" in thal_lower or "freversable defect" in thal_lower or thal_lower in ["7", "7.0"]:
        thal_val = 7.0
    elif "normal" in thal_lower or thal_lower in ["2.31", "2.310"]:
        thal_val = 2.31
    else:
        raise ValueError(f"Invalid thal condition '{thal}'. Please select a valid thal option.")

    # Resting ECG (restecg)
    restecg_str = str(restecg).strip()
    restecg_lower = restecg_str.lower()
    if "nothing to note" in restecg_lower or restecg_lower in ["0", "0.0"]:
        restecg_val = 0.0
    elif "st-t" in restecg_lower or restecg_lower in ["1", "1.0"]:
        restecg_val = 1.0
    elif "hypertrophy" in restecg_lower or restecg_lower in ["2", "2.0"]:
        restecg_val = 2.0
    else:
        raise ValueError(f"Invalid resting electrocardiographic result '{restecg}'. Please select a valid ECG option.")

    # Return validated features in the exact expected feature order (13 features)
    return (
        age_val,
        sex_val,
        cp_val,
        trestbps_val,
        restecg_val,
        chol_val,
        fbs_val,
        thalach_val,
        exang_val,
        oldpeak_val,
        slope_val,
        ca_val,
        thal_val
    )

def predict_from_validated(validated_features):
    """Fast-path prediction from pre-validated numeric features tuple."""
    user_input = np.array(validated_features, dtype=float).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = clfr.predict(user_input)
    return int(prediction[0])

def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):
    validated_features = validate_and_convert_features(
        age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal
    )
    return predict_from_validated(validated_features)
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Attempt to import TensorFlow/Keras, but provide graceful fallback if unavailable
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False


class SimpleANNPredictor:
    """Lightweight, 100% portable neural network predictor using pure NumPy.
    Guarantees exact identical predictions on any Python or Keras version
    without deserialization or dependency mismatch errors.
    """
    def __init__(self, weights):
        # weights: [w0 (29, 16), b0 (16,), w1 (16, 1), b1 (1,)]
        self.w0 = weights[0]
        self.b0 = weights[1]
        self.w1 = weights[2]
        self.b1 = weights[3]

    def predict(self, X, verbose=0):
        # Layer 1: Dense 16 with ReLU activation
        hidden = np.maximum(0, np.dot(X, self.w0) + self.b0)
        # Layer 2: Dense 1 with Linear activation
        output = np.dot(hidden, self.w1) + self.b1
        return output


# Set page title
st.set_page_config(page_title="Chemical Reaction Yield Predictor", layout="centered")

st.title("Chemical Reaction Yield Prediction")
st.write(
    "This web application predicts the chemical reaction yield (%) for the Claisen rearrangement "
    "reaction of catechol diallyl ether using a simple Artificial Neural Network (ANN) trained on "
    "experimental continuous-flow reactor data."
)

# Robust paths to trained model artifacts (works from root or frontend/ directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if os.path.exists(os.path.join(CURRENT_DIR, "..", "models")) else os.path.abspath(".")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ann_model.keras")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models", "ann_weights.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")


@st.cache_resource
def load_artifacts():
    """Load model and preprocessor with multi-tier fallback for 100% cross-version compatibility."""
    if not os.path.exists(PREPROCESSOR_PATH):
        return None, None

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = None

    # Tier 1: Try standard Keras load_model
    if KERAS_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
        except Exception:
            model = None

    # Tier 2: Reconstruct Keras architecture and load weights
    if model is None and KERAS_AVAILABLE and os.path.exists(WEIGHTS_PATH):
        try:
            weights = joblib.load(WEIGHTS_PATH)
            m = keras.Sequential([
                keras.layers.Input(shape=(29,)),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(1, activation='linear')
            ])
            m.compile(optimizer='adam', loss='mean_squared_error')
            m.set_weights(weights)
            model = m
        except Exception:
            model = None

    # Tier 3: Pure NumPy forward-pass predictor (guaranteed to work on any Python version)
    if model is None and os.path.exists(WEIGHTS_PATH):
        try:
            weights = joblib.load(WEIGHTS_PATH)
            model = SimpleANNPredictor(weights)
        except Exception:
            model = None

    return model, preprocessor


model, preprocessor = load_artifacts()

if model is None or preprocessor is None:
    st.error(
        f"Model or preprocessor artifact not found in '{os.path.join(PROJECT_ROOT, 'models')}'. "
        "Please run 'Chemical_Yield_Prediction.ipynb' in the 'training/' folder first."
    )
    st.stop()

# Valid solvent pairs represented in the experimental dataset
SOLVENT_PAIRS = {
    "Methanol": "Ethylene Glycol [1,2-Ethanediol]",
    "1,1,1,3,3,3-Hexafluoropropan-2-ol": "2-Methyltetrahydrofuran [2-MeTHF]",
    "Cyclohexane": "IPA [Propan-2-ol]",
    "Water.Acetonitrile": "Acetonitrile",
    "Acetonitrile": "Acetonitrile.Acetic Acid",
    "2-Methyltetrahydrofuran [2-MeTHF]": "Diethyl Ether [Ether]",
    "2,2,2-Trifluoroethanol": "Water.2,2,2-Trifluoroethanol",
    "DMA [N,N-Dimethylacetamide]": "Decanol",
    "Ethanol": "THF [Tetrahydrofuran]",
    "Dihydrolevoglucosenone (Cyrene)": "Ethyl Acetate",
    "MTBE [tert-Butylmethylether]": "Butanone [MEK]",
    "tert-Butanol [2-Methylpropan-2-ol]": "Dimethyl Carbonate",
    "Methyl Propionate": "Ethyl Lactate",
}

st.header("Reaction Process Parameters")

# Operational inputs
temperature = st.number_input(
    "Temperature (°C)",
    min_value=175.0,
    max_value=225.0,
    value=200.0,
    step=1.0,
    help="Reactor operating temperature in degrees Celsius (Dataset range: 175°C - 225°C)."
)

residence_time = st.number_input(
    "Residence Time (minutes)",
    min_value=2.0,
    max_value=15.0,
    value=10.0,
    step=0.5,
    help="Duration the reaction mixture spends inside the continuous-flow reactor (Dataset range: 2.0 - 15.0 min)."
)

solvent_a = st.selectbox(
    "Solvent A Name (Primary Solvent)",
    options=list(SOLVENT_PAIRS.keys()),
    index=0
)

# Restrict Solvent B to the pair represented in experimental data
paired_solvent_b = SOLVENT_PAIRS[solvent_a]
solvent_b = st.selectbox(
    "Solvent B Name (Secondary Solvent)",
    options=[paired_solvent_b],
    index=0,
    help="Restricted to the experimentally validated solvent pair corresponding to Solvent A."
)

solvent_b_ratio = st.number_input(
    "Solvent B Fraction (SolventB%)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Fraction of Solvent B in the binary solvent mixture (0.0 to 1.0, representing 0% to 100%)."
)

# Predict button
if st.button("Predict Yield"):
    # Create input DataFrame matching the exact training column names and ordering
    input_data = pd.DataFrame([{
        "Temperature": float(temperature),
        "Residence Time": float(residence_time),
        "SOLVENT A NAME": solvent_a,
        "SOLVENT B NAME": solvent_b,
        "SolventB%": float(solvent_b_ratio)
    }])

    # Preprocess inputs using the fitted ColumnTransformer
    input_processed = preprocessor.transform(input_data)

    # Generate prediction from the trained ANN (works with Keras model or NumPy predictor)
    raw_pred = model.predict(input_processed, verbose=0)[0][0]
    predicted_yield = float(np.clip(raw_pred, 0.0, 100.0))

    # Display prediction result
    st.subheader(f"Predicted Yield: {predicted_yield:.2f}%")
    st.info(
        "Note: This result is a machine learning model estimate based on historical "
        "flow-reactor experiments, not an experimentally verified yield."
    )

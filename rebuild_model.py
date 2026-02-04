"""
Rebuild model with TensorFlow 2.15 compatibility
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

print(f"TensorFlow version: {tf.__version__}")

# Create model with explicit Input layer (compatible format)
print("\nCreating new model architecture...")
model = Sequential([
    Input(shape=(14,)),  # Use Input layer instead of input_shape
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Try to load weights from old model
print("Loading weights from old model...")
try:
    old_model = tf.keras.models.load_model("phishing_model_combined.h5", compile=False)
    weights = old_model.get_weights()
    model.set_weights(weights)
    print("✓ Weights transferred successfully")
except Exception as e:
    print(f"⚠️ Could not load old model: {e}")
    print("Please retrain the model by running: python train_combined_model.py")
    exit(1)

# Compile and save in new format
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nSaving model in compatible format...")
model.save("phishing_model_combined.h5", save_format='h5')

# Test reload
print("\nTesting reload...")
test_model = tf.keras.models.load_model("phishing_model_combined.h5")
print("✓ Model loads successfully!")
print("\nModel summary:")
test_model.summary()

# Verify scaler files exist
if os.path.exists("scaler_mean_combined.npy") and os.path.exists("scaler_scale_combined.npy"):
    print("\n✓ Scaler files present")
    mean = np.load("scaler_mean_combined.npy")
    scale = np.load("scaler_scale_combined.npy")
    print(f"  Scaler shape: mean={mean.shape}, scale={scale.shape}")
else:
    print("\n⚠️ Warning: Scaler files missing!")

print("\n✓ Model is ready for deployment!")

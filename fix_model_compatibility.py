"""
Fix TensorFlow model compatibility by re-saving in compatible format
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

print("Loading old model...")
try:
    # Load the model
    old_model = tf.keras.models.load_model("phishing_model_combined.h5", compile=False)
    
    print(f"Model architecture:")
    old_model.summary()
    
    # Re-save in SavedModel format (more compatible)
    print("\nRe-saving model in compatible format...")
    old_model.save("phishing_model_combined.h5", save_format='h5')
    
    print("✓ Model re-saved successfully!")
    
    # Test loading
    print("\nTesting reload...")
    test_model = tf.keras.models.load_model("phishing_model_combined.h5", compile=False)
    print("✓ Model loads successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    
    # Load weights only and rebuild
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    
    # Create model architecture (same as training)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(14,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Try to load weights
    try:
        old_model = tf.keras.models.load_model("phishing_model_combined.h5", compile=False)
        weights = old_model.get_weights()
        model.set_weights(weights)
        
        # Save in new format
        model.save("phishing_model_combined.h5")
        print("✓ Model rebuilt and saved!")
    except Exception as e2:
        print(f"Rebuild failed: {e2}")

"""
Meta-Learning Model for Battleship AI

This module creates a neural network that learns to dynamically adjust
strategy weights based on the current game state. It allows the AI to
adapt its approach throughout a game for optimum performance.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging
from datetime import datetime

# Directory setup
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'meta_learner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MetaLearner')

# Constants
META_FEATURES = 20  # Number of features in the state representation
META_OUTPUTS = 5  # Number of weight adjustments to output
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_PATH = MODELS_DIR / "meta_learner.h5"
HISTORY_PATH = MODELS_DIR / "meta_learner_history.pkl"


def create_meta_model():
    """
    Create a neural network model that learns to adjust strategy weights
    based on the current game state.

    Returns:
        tf.keras.Model: Compiled meta-learning model
    """
    # Input layer: game state features
    inputs = Input(shape=(META_FEATURES,), name='game_state')

    # Hidden layers
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Output layer: weight adjustments for each strategy
    # Using tanh activation to output values between -1 and 1
    # These are adjustments that will be added to base weights
    outputs = Dense(META_OUTPUTS, activation='tanh', name='weight_adjustments')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic training data for the meta-learner.
    In a real implementation, this would be replaced with actual game data.

    Args:
        n_samples: Number of samples to generate

    Returns:
        tuple: (X, y) training data
    """
    # Generate random game states
    X = np.random.rand(n_samples, META_FEATURES)

    # Generate synthetic labels (weight adjustments)
    y = np.zeros((n_samples, META_OUTPUTS))

    # Early game: Focus on hunt mode with parity
    early_game_mask = X[:, 0] < 0.3  # Progress < 30%
    y[early_game_mask, 0] = 0.8  # High density weight
    y[early_game_mask, 1] = 0.3  # Low neural weight
    y[early_game_mask, 2] = 0.2  # Low Monte Carlo weight

    # Mid game: Balance between strategies
    mid_game_mask = (X[:, 0] >= 0.3) & (X[:, 0] < 0.6)
    y[mid_game_mask, 0] = 0.5  # Medium density weight
    y[mid_game_mask, 1] = 0.5  # Medium neural weight
    y[mid_game_mask, 2] = 0.5  # Medium Monte Carlo weight

    # Late game: Focus on probabilistic simulation
    late_game_mask = X[:, 0] >= 0.6
    y[late_game_mask, 0] = 0.3  # Low density weight
    y[late_game_mask, 1] = 0.4  # Medium neural weight
    y[late_game_mask, 2] = 0.8  # High Monte Carlo weight

    # Targeting mode: Focus on Monte Carlo and less on neural
    targeting_mask = X[:, 5] > 0.5  # Targeting mode active
    y[targeting_mask, 1] = 0.3  # Lower neural weight in targeting mode
    y[targeting_mask, 2] = 0.7  # Higher Monte Carlo in targeting mode

    # Few remaining ships: Information gain becomes more important
    few_ships_mask = (X[:, 12] < 0.4) & (X[:, 0] > 0.7)  # Min ship size < 2 and > 70% complete
    y[few_ships_mask, 3] = 0.9  # Very high information gain weight

    # Add some random noise
    y += np.random.normal(0, 0.1, y.shape)

    # Clip to [-1, 1] range for tanh activation
    y = np.clip(y, -1, 1)

    return X, y


def load_game_data():
    """
    Load real game data to train the meta-learner.

    Returns:
        tuple: (X, y) training data, or None if no data available
    """
    try:
        # Check for saved game states
        game_states_path = DATA_DIR / "game_states.pkl"
        if not game_states_path.exists():
            logger.info("No real game data found. Using synthetic data.")
            return None

        # Load game states
        with open(game_states_path, 'rb') as f:
            game_data = pickle.load(f)

        if not game_data or len(game_data) < 100:
            logger.info("Insufficient real game data. Using synthetic data.")
            return None

        # Extract features and labels
        X = []
        y = []

        for game_id, states in game_data.items():
            for state in states:
                if 'features' in state and 'weights' in state:
                    X.append(state['features'])
                    y.append(state['weights'])

        if len(X) < 100:
            logger.info("Insufficient feature/weight pairs. Using synthetic data.")
            return None

        return np.array(X), np.array(y)

    except Exception as e:
        logger.error(f"Error loading game data: {e}")
        return None


def train_meta_model(model=None, epochs=EPOCHS, batch_size=BATCH_SIZE, use_synthetic=True):
    """
    Train the meta-learning model with either real or synthetic data.

    Args:
        model: Existing model to train, or None to create a new one
        epochs: Number of training epochs
        batch_size: Training batch size
        use_synthetic: Whether to use synthetic data

    Returns:
        tf.keras.Model: Trained model
    """
    # Create model if not provided
    if model is None:
        model = create_meta_model()
        logger.info("Created new meta-learning model")

    # Load real data if available and not using synthetic
    real_data = None if use_synthetic else load_game_data()

    if real_data is not None:
        X, y = real_data
        logger.info(f"Training with {len(X)} real game state samples")
    else:
        # Generate synthetic data
        X, y = generate_synthetic_data()
        logger.info(f"Training with {len(X)} synthetic data samples")

    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            save_best_only=True,
            monitor='val_loss'
        ),
        TensorBoard(log_dir=str(LOG_DIR / f"meta_learner_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(LOG_DIR / "meta_learner_training.png")

    return model


def test_meta_model(model=None):
    """
    Test the meta-learning model with some sample game states.

    Args:
        model: Model to test, or None to load from disk
    """
    # Load model if not provided
    if model is None:
        if not MODEL_PATH.exists():
            logger.error("No saved model found. Train a model first.")
            return

        model = load_model(MODEL_PATH)
        logger.info("Loaded meta-learning model from disk")

    # Create test examples
    test_cases = [
        {"name": "Early Game", "features": np.zeros(META_FEATURES)},
        {"name": "Mid Game", "features": np.zeros(META_FEATURES)},
        {"name": "Late Game", "features": np.zeros(META_FEATURES)},
        {"name": "Targeting Mode", "features": np.zeros(META_FEATURES)},
    ]

    # Set specific test features
    test_cases[0]["features"][0] = 0.1  # Progress = 10%
    test_cases[1]["features"][0] = 0.5  # Progress = 50%
    test_cases[2]["features"][0] = 0.8  # Progress = 80%
    test_cases[3]["features"][0] = 0.4  # Progress = 40%
    test_cases[3]["features"][5] = 1.0  # Targeting mode

    # Run predictions
    for test_case in test_cases:
        features = np.array([test_case["features"]])
        predictions = model.predict(features)[0]

        logger.info(f"Test Case: {test_case['name']}")
        logger.info(f"Predictions: density={predictions[0]:.2f}, neural={predictions[1]:.2f}, " +
                    f"montecarlo={predictions[2]:.2f}, info_gain={predictions[3]:.2f}, " +
                    f"opponent_model={predictions[4]:.2f}")

        # Log result interpretation
        if test_case["name"] == "Early Game":
            if predictions[0] > predictions[2]:
                logger.info("✓ Correctly prioritizes density-based search in early game")
            else:
                logger.info("✗ Fails to prioritize density-based search in early game")
        elif test_case["name"] == "Late Game":
            if predictions[2] > predictions[0]:
                logger.info("✓ Correctly prioritizes Monte Carlo in late game")
            else:
                logger.info("✗ Fails to prioritize Monte Carlo in late game")
        elif test_case["name"] == "Targeting Mode":
            if predictions[2] > predictions[1]:
                logger.info("✓ Correctly prioritizes Monte Carlo over neural in targeting mode")
            else:
                logger.info("✗ Fails to prioritize Monte Carlo over neural in targeting mode")


def run_meta_learner_training():
    """Main function to train and test the meta-learner"""
    logger.info("Starting meta-learner training...")

    # Check if we should retrain or use existing model
    retrain = True
    if MODEL_PATH.exists():
        response = input("Meta-learner model already exists. Retrain? (y/n): ")
        retrain = response.lower() == 'y'

    if retrain:
        # Train with synthetic data
        model = train_meta_model(use_synthetic=True)
        logger.info(f"Meta-learner model trained and saved to {MODEL_PATH}")
    else:
        # Load existing model
        model = load_model(MODEL_PATH)
        logger.info(f"Loaded existing meta-learner model from {MODEL_PATH}")

    # Test the model
    test_meta_model(model)


if __name__ == "__main__":
    run_meta_learner_training()
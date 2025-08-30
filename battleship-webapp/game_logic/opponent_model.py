"""
Opponent Modeling Network for Battleship AI

This module creates a neural network that learns to predict opponent ship placements
and targeting strategies based on observed patterns. It allows the AI to adapt its
approach based on the specific opponent it's facing.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

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
        logging.FileHandler(LOG_DIR / 'opponent_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OpponentModel')

# Constants
BOARD_SIZE = 10
INPUT_CHANNELS = 5  # Board state planes: misses, hits, unknown, opponent_tendency, edge
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_PATH = MODELS_DIR / "opponent_model.h5"
HISTORY_PATH = MODELS_DIR / "opponent_model_history.pkl"
PROFILES_PATH = MODELS_DIR / "opponent_profiles.pkl"


def create_opponent_model():
    """
    Create a neural network model that predicts opponent ship placements.
    The model takes the current board state and outputs a heatmap of likely
    ship locations.

    Returns:
        tf.keras.Model: Compiled opponent modeling network
    """
    # Input layer: board state with multiple channels
    inputs = Input(shape=(BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS), name='board_state')

    # Convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Output convolution to produce heatmap
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='ship_heatmap')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def generate_synthetic_opponent_data(n_samples=1000, n_opponents=5):
    """
    Generate synthetic training data for opponent modeling.
    In a real implementation, this would be replaced with actual game data.

    Args:
        n_samples: Number of samples to generate per opponent
        n_opponents: Number of different opponent types to simulate

    Returns:
        tuple: (X, y) training data
    """
    X = []
    y = []

    # Create different opponent placement patterns
    opponent_patterns = []

    # 1. Edge-loving opponent
    edge_pattern = np.zeros((BOARD_SIZE, BOARD_SIZE))
    edge_pattern[0, :] = 1.0
    edge_pattern[-1, :] = 1.0
    edge_pattern[:, 0] = 1.0
    edge_pattern[:, -1] = 1.0
    opponent_patterns.append(edge_pattern)

    # 2. Corner-loving opponent
    corner_pattern = np.zeros((BOARD_SIZE, BOARD_SIZE))
    corner_size = 3
    corner_pattern[:corner_size, :corner_size] = 1.0
    corner_pattern[:corner_size, -corner_size:] = 1.0
    corner_pattern[-corner_size:, :corner_size] = 1.0
    corner_pattern[-corner_size:, -corner_size:] = 1.0
    opponent_patterns.append(corner_pattern)

    # 3. Center-loving opponent
    center_pattern = np.zeros((BOARD_SIZE, BOARD_SIZE))
    center_radius = 3
    center_r, center_c = BOARD_SIZE // 2, BOARD_SIZE // 2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if abs(r - center_r) + abs(c - center_c) <= center_radius:
                center_pattern[r, c] = 1.0
    opponent_patterns.append(center_pattern)

    # 4. Diagonal-loving opponent
    diag_pattern = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(BOARD_SIZE):
        diag_pattern[i, i] = 1.0
        diag_pattern[i, BOARD_SIZE - i - 1] = 1.0
    opponent_patterns.append(diag_pattern)

    # 5. Random (uniform) opponent
    uniform_pattern = np.ones((BOARD_SIZE, BOARD_SIZE))
    opponent_patterns.append(uniform_pattern)

    # Normalize patterns
    for i in range(len(opponent_patterns)):
        opponent_patterns[i] = opponent_patterns[i] / opponent_patterns[i].sum()

    # Generate samples for each opponent
    for opponent_idx in range(n_opponents):
        pattern = opponent_patterns[opponent_idx]

        for _ in range(n_samples):
            # Generate random board state
            miss_plane = np.random.binomial(1, 0.2, (BOARD_SIZE, BOARD_SIZE))
            hit_plane = np.random.binomial(1, 0.1, (BOARD_SIZE, BOARD_SIZE))
            unknown_plane = 1.0 - (miss_plane + hit_plane)

            # Ensure no overlaps
            hit_plane = hit_plane * (1.0 - miss_plane)
            unknown_plane = unknown_plane * (1.0 - miss_plane) * (1.0 - hit_plane)

            # Generate opponent tendency plane (would be learned from past games)
            tendency_plane = pattern + np.random.normal(0, 0.05, pattern.shape)
            tendency_plane = np.clip(tendency_plane, 0, 1)

            # Edge plane (fixed)
            edge_plane = np.zeros((BOARD_SIZE, BOARD_SIZE))
            edge_plane[0, :] = 1.0
            edge_plane[-1, :] = 1.0
            edge_plane[:, 0] = 1.0
            edge_plane[:, -1] = 1.0

            # Stack channels to create input
            board_state = np.stack([miss_plane, hit_plane, unknown_plane, tendency_plane, edge_plane], axis=-1)

            # Create target (expected ship placement)
            # Sample from opponent pattern with some noise
            target = np.zeros((BOARD_SIZE, BOARD_SIZE, 1))
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if miss_plane[r, c] == 0 and hit_plane[r, c] == 0:  # Only place ships on unknown squares
                        target[r, c, 0] = np.random.binomial(1, pattern[r, c])

            X.append(board_state)
            y.append(target)

    return np.array(X), np.array(y)


def load_opponent_data():
    """
    Load real opponent data from past games.

    Returns:
        tuple: (X, y) training data, or None if no data available
    """
    try:
        # Check for saved opponent data
        opponent_data_path = DATA_DIR / "opponent_data.pkl"
        if not opponent_data_path.exists():
            logger.info("No real opponent data found. Using synthetic data.")
            return None

        # Load opponent data
        with open(opponent_data_path, 'rb') as f:
            opponent_data = pickle.load(f)

        if not opponent_data or len(opponent_data) < 50:
            logger.info("Insufficient real opponent data. Using synthetic data.")
            return None

        # Extract features and labels
        X = []
        y = []

        for game_id, data in opponent_data.items():
            if 'board_states' in data and 'ship_placements' in data:
                X.extend(data['board_states'])
                y.extend(data['ship_placements'])

        if len(X) < 50:
            logger.info("Insufficient board state/ship placement pairs. Using synthetic data.")
            return None

        return np.array(X), np.array(y)

    except Exception as e:
        logger.error(f"Error loading opponent data: {e}")
        return None


def load_opponent_profiles():
    """
    Load saved opponent profiles.

    Returns:
        dict: Opponent profiles, or empty dict if none found
    """
    if not PROFILES_PATH.exists():
        return {}

    try:
        with open(PROFILES_PATH, 'rb') as f:
            profiles = pickle.load(f)
        return profiles
    except Exception as e:
        logger.error(f"Error loading opponent profiles: {e}")
        return {}


def save_opponent_profiles(profiles):
    """
    Save opponent profiles.

    Args:
        profiles: Opponent profiles to save
    """
    try:
        with open(PROFILES_PATH, 'wb') as f:
            pickle.dump(profiles, f)
        logger.info("Opponent profiles saved successfully")
    except Exception as e:
        logger.error(f"Error saving opponent profiles: {e}")


def train_opponent_model(model=None, epochs=EPOCHS, batch_size=BATCH_SIZE, use_synthetic=True):
    """
    Train the opponent modeling network with either real or synthetic data.

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
        model = create_opponent_model()
        logger.info("Created new opponent modeling network")

    # Load real data if available and not using synthetic
    real_data = None if use_synthetic else load_opponent_data()

    if real_data is not None:
        X, y = real_data
        logger.info(f"Training with {len(X)} real opponent data samples")
    else:
        # Generate synthetic data
        X, y = generate_synthetic_opponent_data()
        logger.info(f"Training with {len(X)} synthetic opponent data samples")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            save_best_only=True,
            monitor='val_loss'
        ),
        TensorBoard(log_dir=str(LOG_DIR / f"opponent_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
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
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(LOG_DIR / "opponent_model_training.png")

    return model


def visualize_opponent_predictions(model=None):
    """
    Visualize opponent ship placement predictions for different opponent types.

    Args:
        model: Model to use, or None to load from disk
    """
    # Load model if not provided
    if model is None:
        if not MODEL_PATH.exists():
            logger.error("No saved model found. Train a model first.")
            return

        model = load_model(MODEL_PATH)
        logger.info("Loaded opponent model from disk")

    # Create test board states
    test_board_states = []
    test_names = ["Edge Opponent", "Corner Opponent", "Center Opponent", "Diagonal Opponent", "Random Opponent"]

    # Base board state - 20% misses, 10% hits, rest unknown
    base_miss = np.random.binomial(1, 0.2, (BOARD_SIZE, BOARD_SIZE))
    base_hit = np.random.binomial(1, 0.1, (BOARD_SIZE, BOARD_SIZE))
    base_hit = base_hit * (1.0 - base_miss)  # Ensure no overlaps
    base_unknown = 1.0 - (base_miss + base_hit)

    # Edge plane (fixed)
    edge_plane = np.zeros((BOARD_SIZE, BOARD_SIZE))
    edge_plane[0, :] = 1.0
    edge_plane[-1, :] = 1.0
    edge_plane[:, 0] = 1.0
    edge_plane[:, -1] = 1.0

    # Create different opponent tendency planes
    # 1. Edge-loving opponent
    edge_tendency = np.zeros((BOARD_SIZE, BOARD_SIZE))
    edge_tendency[0, :] = 1.0
    edge_tendency[-1, :] = 1.0
    edge_tendency[:, 0] = 1.0
    edge_tendency[:, -1] = 1.0
    edge_tendency = edge_tendency / edge_tendency.sum()

    # 2. Corner-loving opponent
    corner_tendency = np.zeros((BOARD_SIZE, BOARD_SIZE))
    corner_size = 3
    corner_tendency[:corner_size, :corner_size] = 1.0
    corner_tendency[:corner_size, -corner_size:] = 1.0
    corner_tendency[-corner_size:, :corner_size] = 1.0
    corner_tendency[-corner_size:, -corner_size:] = 1.0
    corner_tendency = corner_tendency / corner_tendency.sum()

    # 3. Center-loving opponent
    center_tendency = np.zeros((BOARD_SIZE, BOARD_SIZE))
    center_radius = 3
    center_r, center_c = BOARD_SIZE // 2, BOARD_SIZE // 2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if abs(r - center_r) + abs(c - center_c) <= center_radius:
                center_tendency[r, c] = 1.0
    center_tendency = center_tendency / center_tendency.sum()

    # 4. Diagonal-loving opponent
    diag_tendency = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(BOARD_SIZE):
        diag_tendency[i, i] = 1.0
        diag_tendency[i, BOARD_SIZE - i - 1] = 1.0
    diag_tendency = diag_tendency / diag_tendency.sum()

    # 5. Random (uniform) opponent
    uniform_tendency = np.ones((BOARD_SIZE, BOARD_SIZE))
    uniform_tendency = uniform_tendency / uniform_tendency.sum()

    # Create test board states
    tendencies = [edge_tendency, corner_tendency, center_tendency, diag_tendency, uniform_tendency]

    for tendency in tendencies:
        board_state = np.stack([base_miss, base_hit, base_unknown, tendency, edge_plane], axis=-1)
        test_board_states.append(board_state)

    # Make predictions
    test_board_states = np.array(test_board_states)
    predictions = model.predict(test_board_states)

    # Plot predictions
    plt.figure(figsize=(15, 10))

    for i, (name, prediction) in enumerate(zip(test_names, predictions)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(prediction[:, :, 0], cmap='viridis')
        plt.title(name)
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(LOG_DIR / "opponent_model_predictions.png")
    logger.info(f"Opponent model predictions visualized and saved to {LOG_DIR / 'opponent_model_predictions.png'}")


def update_opponent_profile(opponent_id, game_data):
    """
    Update an opponent profile based on observed game data.

    Args:
        opponent_id: Identifier for the opponent
        game_data: Data from the game
    """
    # Load existing profiles
    profiles = load_opponent_profiles()

    # Create new profile if needed
    if opponent_id not in profiles:
        profiles[opponent_id] = {
            'placement_tendencies': {},
            'attack_patterns': {},
            'ship_orientations': {'horizontal': 0, 'vertical': 0},
            'clustering_tendency': 0.5,
            'games_observed': 0
        }

    # Update profile based on game data
    if 'ship_placements' in game_data:
        for r, c in game_data['ship_placements']:
            key = f"{r}-{c}"
            if key in profiles[opponent_id]['placement_tendencies']:
                profiles[opponent_id]['placement_tendencies'][key] += 1
            else:
                profiles[opponent_id]['placement_tendencies'][key] = 1

    if 'moves' in game_data:
        for r, c in game_data['moves']:
            key = f"{r}-{c}"
            if key in profiles[opponent_id]['attack_patterns']:
                profiles[opponent_id]['attack_patterns'][key] += 1
            else:
                profiles[opponent_id]['attack_patterns'][key] = 1

    if 'ship_orientations' in game_data:
        for orientation, count in game_data['ship_orientations'].items():
            profiles[opponent_id]['ship_orientations'][orientation] += count

    if 'clustering_score' in game_data:
        old_score = profiles[opponent_id]['clustering_tendency']
        old_games = profiles[opponent_id]['games_observed']
        new_score = (old_score * old_games + game_data['clustering_score']) / (old_games + 1)
        profiles[opponent_id]['clustering_tendency'] = new_score

    # Increment games observed
    profiles[opponent_id]['games_observed'] += 1

    # Normalize tendencies
    for category in ['placement_tendencies', 'attack_patterns']:
        total = sum(profiles[opponent_id][category].values())
        if total > 0:
            for key in profiles[opponent_id][category]:
                profiles[opponent_id][category][key] /= total

    # Save updated profiles
    save_opponent_profiles(profiles)
    logger.info(f"Updated profile for opponent {opponent_id}")


def run_opponent_model_training():
    """Main function to train and visualize the opponent model"""
    logger.info("Starting opponent model training...")

    # Check if we should retrain or use existing model
    retrain = True
    if MODEL_PATH.exists():
        response = input("Opponent model already exists. Retrain? (y/n): ")
        retrain = response.lower() == 'y'

    if retrain:
        # Train with synthetic data
        model = train_opponent_model(use_synthetic=True)
        logger.info(f"Opponent model trained and saved to {MODEL_PATH}")
    else:
        # Load existing model
        model = load_model(MODEL_PATH)
        logger.info(f"Loaded existing opponent model from {MODEL_PATH}")

    # Visualize predictions
    visualize_opponent_predictions(model)


if __name__ == "__main__":
    run_opponent_model_training()
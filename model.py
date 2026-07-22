from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
import numpy as np

def build_classifier(input_shape, n_horizons, units=64, dropout=0.4):
    """
    IMPROVED ARCHITECTURE:
    - Bidirectional LSTM (looks forward AND backward in time)
    - Stacked LSTMs (2 layers to capture complex patterns)
    - Batch Normalization (faster convergence)
    - L2 Regularization (combat overfitting)
    - Higher dropout (0.4 instead of 0.3)
    """
    model = Sequential([
        # First LSTM layer (with return_sequences=True to stack)
        Bidirectional(LSTM(units, return_sequences=True, 
                          kernel_regularizer=l2(0.001)), 
                     input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        
        # Second LSTM layer
        Bidirectional(LSTM(units // 2, kernel_regularizer=l2(0.001))),
        BatchNormalization(),
        Dropout(dropout),
        
        # Dense layers for feature combination
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout / 2),
        
        # Output
        Dense(n_horizons, activation='sigmoid')
    ])
    
    print(f"✓ Built Bidirectional Stacked LSTM: {units}→{units//2}→32→16→{n_horizons}")
    return model

def compile_model(model, learning_rate=0.0005):
    """Lower learning rate for stable convergence"""
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Gradient clipping
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(f"✓ Model compiled (LR={learning_rate}, Gradient Clipping enabled)")
    return model

def get_callbacks(patience=20):
    """
    IMPROVED CALLBACKS:
    - Longer patience (20 epochs) to let model learn
    - More aggressive LR reduction
    """
    return [
        EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True, 
            verbose=1,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3,  # More aggressive reduction (0.3 instead of 0.5)
            patience=10, 
            verbose=1,
            min_lr=0.00001
        )
    ]

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, callbacks=None):
    """
    CRITICAL CHANGE: shuffle=True in training
    Reason: Breaking temporal correlation in batches helps generalization
    (We still split data temporally, but shuffle within training set)
    """
    y_flat = y_train.flatten()
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_flat)
    class_weights_dict = dict(enumerate(weights))
    
    print(f"Training with class weights: {class_weights_dict}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,  # CHANGED: Shuffle to break temporal correlation in batches
        verbose=1,     # Changed to 1 to see progress
        class_weight=class_weights_dict
    )
    return history
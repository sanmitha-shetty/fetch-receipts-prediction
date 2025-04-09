
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time 

from src.data_processor import load_data, create_features, scale_features

DATA_FILEPATH = 'data/data_daily.csv'
CHECKPOINT_SAVE_DIR = 'models/tf_checkpoint' # directory
SCALER_FILENAME = 'scaler_params.npz' 
FEATURE_COLS_TO_DROP = ['Date', 'Receipt_Count', 'year']
TARGET_COL = 'Receipt_Count'

LEARNING_RATE = 0.0005 
EPOCHS = 350         
BATCH_SIZE = 32
HIDDEN_UNITS = 128     


class SimpleMLP(tf.keras.Model):
    def __init__(self, num_features, hidden_units):
        super(SimpleMLP, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu', name='dense_1',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout1 = tf.keras.layers.Dropout(0.2) 
        
        self.dense2 = tf.keras.layers.Dense(hidden_units // 2, activation='relu', name='dense_2',
                                             kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout2 = tf.keras.layers.Dropout(0.15) 
        self.dense3 = tf.keras.layers.Dense(1, name='output') 
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training) 
        return self.dense3(x)

# MSE 
mse_loss = tf.keras.losses.MeanSquaredError()

#use Adam with the adjusted learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(model, inputs, targets_log):
    with tf.GradientTape() as tape:
        predictions_log = model(inputs, training=True)
        targets_log = tf.reshape(targets_log, [-1, 1]) 
        loss_value = mse_loss(targets_log, predictions_log)
        # Add regularization losses 
        loss_value += sum(model.losses)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value 

def train_tf_model():
    """Loads data, preprocesses, trains the TensorFlow MLP model, and saves it."""
    print("--- Starting TensorFlow Model Training (MSE on log1p(Target)) ---")
    start_time = time.time()

    try:
        df = load_data(DATA_FILEPATH)
    except Exception as e:
        print(f"Failed to load data. Exiting. Error: {e}")
        return

    df_featured = create_features(df)
    
    feature_columns = [col for col in df_featured.columns if col not in FEATURE_COLS_TO_DROP]
    print(f"Using {len(feature_columns)} features.")
    print(f"Feature columns sample: {sorted(feature_columns)[:15]}...") 

    X = df_featured[feature_columns].values 
    y_original = df_featured[TARGET_COL].values 

    # LOG1Ptransformation
    y_log = np.log1p(y_original)
    if np.any(np.isnan(y_log)) or np.any(np.isinf(y_log)):
        print("Warning: NaN or Inf detected in log-transformed target. Check original data.")
    print(f"Target variable '{TARGET_COL}' transformed using np.log1p.")
    print(f"Sample log-transformed y: {y_log[:5]}")

    X_scaled, _, scaler_params = scale_features(X) 
    num_features = X_scaled.shape[1]
    print(f"Features scaled. Shape: {X_scaled.shape}, Data type: {X_scaled.dtype}")


    dataset = tf.data.Dataset.from_tensor_slices((X_scaled, y_log.astype(np.float32)))
    dataset = dataset.shuffle(buffer_size=len(X_scaled)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print("TensorFlow dataset created.")

    model = SimpleMLP(num_features=num_features, hidden_units=HIDDEN_UNITS)
    _ = model(tf.zeros((1, num_features), dtype=tf.float32))
    model.summary()

    os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model) 
    ckpt_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_SAVE_DIR, max_to_keep=1) 
    print(f"Checkpoint manager configured to save in '{CHECKPOINT_SAVE_DIR}'.")

    print(f"\nStarting training for {EPOCHS} epochs...")
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        total_epoch_loss = 0.0
        num_batches = 0

        for batch_x, batch_y_log in dataset:
            batch_loss = train_step(model, batch_x, batch_y_log)
            total_epoch_loss += batch_loss.numpy() 
            num_batches += 1

        avg_epoch_loss = total_epoch_loss / num_batches
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{EPOCHS} - Avg MSE(log) Loss: {avg_epoch_loss:.6f} - Duration: {epoch_duration:.2f}s")

        if avg_epoch_loss < best_loss:
            print(f"Loss improved from {best_loss:.6f} to {avg_epoch_loss:.6f}. Saving checkpoint.")
            best_loss = avg_epoch_loss
            save_path = ckpt_manager.save()
            print(f"Checkpoint saved: {save_path}")

    print("\nTraining finished.")

    scaler_path = os.path.join(CHECKPOINT_SAVE_DIR, SCALER_FILENAME)
    try:
        np.savez(scaler_path,
                 scaler_mean=scaler_params['mean'],
                 scaler_std=scaler_params['std'],
                 feature_columns=np.array(feature_columns)) 
        print(f"Scaler parameters and feature columns saved to: {scaler_path}")
    except Exception as e:
        print(f"Error saving scaler parameters: {e}")


    total_training_time = time.time() - start_time
    print(f"--- Total Training Time: {total_training_time:.2f} seconds ---")

if __name__ == "__main__":
    # np.random.seed(42)
    # tf.random.set_seed(42)
    train_tf_model()
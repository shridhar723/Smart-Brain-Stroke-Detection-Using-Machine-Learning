import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Dataset directory
dataset_dir = r"D:\project2\training_set"

# Load 70% training data
train_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.3,
    subset="training",
    seed=123,
    label_mode='int'
)

# Load 30% validation data
val_dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.3,
    subset="validation",
    seed=123,
    label_mode='int'
)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(128, 128, 3)),  # Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),  # First Conv Layer
    layers.MaxPooling2D((2, 2)),  # Max Pooling Layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second Conv Layer
    layers.MaxPooling2D((2, 2)),  # Max Pooling Layer
    layers.Conv2D(128, (3, 3), activation='relu'),  # Third Conv Layer
    layers.MaxPooling2D((2, 2)),  # Max Pooling Layer
    layers.Flatten(),  # Flatten layer
    layers.Dense(128, activation='relu'),  # Dense layer with ReLU
    layers.Dense(1, activation='sigmoid')  # Output layer with Sigmoid (binary classification)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary Crossentropy for binary classification
              metrics=['accuracy'])  # Metric to track accuracy

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save the trained model
model.save("brain_stroke_detection_model.h5")

# Plot accuracy for training vs validation
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

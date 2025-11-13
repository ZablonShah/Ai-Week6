
"""
edge_ai_recycle.py
Lightweight image classifier for recyclable items + TFLite conversion.
Run:
    python edge_ai_recycle.py --data_dir path/to/data --epochs 5
"""
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape=(128,128,3), num_classes=5):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main(data_dir, epochs):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(128,128), subset='training', batch_size=32)
    val_gen = datagen.flow_from_directory(data_dir, target_size=(128,128), subset='validation', batch_size=32)

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save Keras model
    model.save('recycle_model.h5')
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('recycle_model.tflite', 'wb').write(tflite_model)
    print("Saved: recycle_model.h5 and recycle_model.tflite")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args.data_dir, args.epochs)

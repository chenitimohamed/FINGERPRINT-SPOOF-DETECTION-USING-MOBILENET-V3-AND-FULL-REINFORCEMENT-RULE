import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import json
import cv2
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

class FROFusion:
    """FRO Triple Π Combination Rule Fusion"""
    
    @staticmethod
    def triple_pi_fusion(scores):
        """
        FRO Triple Π Combination Rule
        Π(s₁,s₂,…,sₙ) = (∏ᵢ₌₁ⁿ Sᵢ) / (∏ᵢ₌₁ⁿ Sᵢ + ∏ᵢ₌₁ⁿ S̄ᵢ)
        where S̄ = 1 - S
        """
        scores = np.array(scores)
        
        # Avoid numerical instability
        scores = np.clip(scores, 1e-10, 1.0 - 1e-10)
        
        # Calculate products
        product_s = np.prod(scores)  # ∏ Sᵢ
        product_complement = np.prod(1 - scores)  # ∏ S̄ᵢ
        
        # FRO fusion formula
        fused_score = product_s / (product_s + product_complement)
        
        return float(fused_score)

class MobileNetV3Fingerprint:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build MobileNetV3 model for fingerprint spoof detection"""
        # Load pre-trained MobileNetV3
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        # Add custom head for fingerprint spoof detection
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile_model(self):
        """Compile the model with appropriate settings"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def train_patch_model(self, train_data, val_data, epochs=50):
        """Train the model on patch-level data"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                'best_patch_model.h5', save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

class FingerprintSpoofDetector:
    def __init__(self, patch_model_path=None):
        self.patch_model = None
        self.fusion_rule = FROFusion()
        
        if patch_model_path and os.path.exists(patch_model_path):
            self.load_patch_model(patch_model_path)
    
    def load_patch_model(self, model_path):
        """Load pre-trained patch-level model"""
        self.patch_model = tf.keras.models.load_model(model_path)
        print("Patch model loaded successfully")
    
    def predict_patch(self, patch):
        """Predict spoofness score for a single patch"""
        if self.patch_model is None:
            raise ValueError("Patch model not loaded. Train or load a model first.")
        
        # Preprocess patch
        patch_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(
            patch[np.newaxis, ...]
        )
        
        # Get prediction
        prediction = self.patch_model.predict(patch_preprocessed, verbose=0)
        spoof_score = prediction[0][1]  # Probability of being spoof
        
        return spoof_score
    
    def predict_image(self, patch_paths):
        """
        Predict spoofness for an entire image using FRO fusion
        patch_paths: list of paths to patches from the same fingerprint image
        """
        patch_scores = []
        
        for patch_path in patch_paths:
            # Load and preprocess patch
            patch = cv2.imread(patch_path)
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = cv2.resize(patch, (224, 224))
            
            # Get patch score
            patch_score = self.predict_patch(patch)
            patch_scores.append(patch_score)
        
        # Apply FRO fusion rule
        global_score = self.fusion_rule.triple_pi_fusion(patch_scores)
        
        return global_score, patch_scores

def create_data_generator(metadata_path, batch_size=32):
    """Create data generators from patch metadata"""
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metadata)
    
    # Filter for training data
    train_df = df[df['split'] == 'Training']
    test_df = df[df['split'] == 'Testing']
    
    # Create image data generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='patch_path',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='patch_path',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='patch_path',
        y_col='class',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, df

def evaluate_complete_system(detector, test_df):
    """Evaluate the complete system with FRO fusion"""
    
    # Group patches by original image
    image_groups = test_df.groupby('original_image')
    
    results = []
    
    for image_name, group in image_groups:
        patch_paths = group['patch_path'].tolist()
        true_label = group['class'].iloc[0]  # All patches from same image have same label
        
        # Get global spoofness score using FRO fusion
        global_score, patch_scores = detector.predict_image(patch_paths)
        
        results.append({
            'image_name': image_name,
            'true_label': true_label,
            'global_score': global_score,
            'predicted_label': 'spoof' if global_score > 0.5 else 'live',
            'num_patches': len(patch_paths),
            'patch_scores_mean': np.mean(patch_scores),
            'patch_scores_std': np.std(patch_scores)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = (results_df['true_label'] == 'spoof').astype(int)
    y_pred = (results_df['predicted_label'] == 'spoof').astype(int)
    y_scores = results_df['global_score']
    
    print("=== Complete System Evaluation with FRO Fusion ===")
    print(classification_report(y_true, y_pred, target_names=['live', 'spoof']))
    print(f"AUC Score: {roc_auc_score(y_true, y_scores):.4f}")
    
    return results_df

def main():
    """Main training and evaluation pipeline"""
    
    # Paths
    metadata_path = r"C:\Users\Mohamed\livdet2015_patches\patch_metadata.json"
    
    # Step 1: Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen, full_df = create_data_generator(metadata_path)
    
    # Step 2: Build and train patch-level model
    print("Building MobileNetV3 model...")
    fingerprint_model = MobileNetV3Fingerprint()
    model = fingerprint_model.build_model()
    fingerprint_model.compile_model()
    
    print("Training patch-level model...")
    history = fingerprint_model.train_patch_model(train_gen, val_gen, epochs=50)
    
    # Step 3: Create complete spoof detector with FRO fusion
    print("Creating complete spoof detector...")
    detector = FingerprintSpoofDetector('best_patch_model.h5')
    
    # Step 4: Evaluate complete system
    print("Evaluating complete system with FRO fusion...")
    test_df = full_df[full_df['split'] == 'Testing']
    results = evaluate_complete_system(detector, test_df)
    
    # Step 5: Save results
    results.to_csv('fro_fusion_results.csv', index=False)
    print("Results saved to fro_fusion_results.csv")
    
    return detector, results

if __name__ == "__main__":
    # Configure GPU for RTX 3090
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    detector, results = main()
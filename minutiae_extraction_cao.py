import os
import cv2
import numpy as np
import math
from scipy import ndimage
from scipy.signal import convolve2d
import pandas as pd
import json

class CaoMinutiaeExtractor:
    def __init__(self, target_minutiae_count=46):
        self.target_minutiae_count = target_minutiae_count
        
    def standardize_to_500dpi(self, image, original_dpi=1000):
        """Standardize image to 500 dPI as per paper requirements"""
        if original_dpi != 500:
            scale_factor = 500.0 / original_dpi
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            standardized_image = cv2.resize(image, (new_width, new_height), 
                                          interpolation=cv2.INTER_AREA)
            return standardized_image
        return image
    
    def enhance_image(self, image):
        """Fingerprint enhancement using Gabor filters"""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(normalized)
        return enhanced
    
    def compute_orientation_map(self, image, block_size=16):
        """Compute ridge orientation map"""
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        orientation = np.zeros_like(image, dtype=np.float64)
        
        for i in range(0, image.shape[0]-block_size, block_size):
            for j in range(0, image.shape[1]-block_size, block_size):
                Gxx = np.sum(gx[i:i+block_size, j:j+block_size] ** 2)
                Gyy = np.sum(gy[i:i+block_size, j:j+block_size] ** 2)
                Gxy = np.sum(gx[i:i+block_size, j:j+block_size] * gy[i:i+block_size, j:j+block_size])
                
                theta = 0.5 * math.atan2(2 * Gxy, Gxx - Gyy) if (Gxx - Gyy) != 0 else 0
                orientation[i:i+block_size, j:j+block_size] = theta
        
        return orientation
    
    def binarize_image(self, image, orientation_map):
        """Adaptive binarization based on local ridge orientation"""
        binary = np.zeros_like(image, dtype=np.uint8)
        block_size = 16
        
        for i in range(0, image.shape[0]-block_size, block_size):
            for j in range(0, image.shape[1]-block_size, block_size):
                block = image[i:i+block_size, j:j+block_size]
                theta = np.mean(orientation_map[i:i+block_size, j:j+block_size])
                
                filter_size = 5
                oriented_filter = self.create_oriented_gaussian(filter_size, theta)
                filtered_block = convolve2d(block, oriented_filter, mode='same', boundary='symm')
                
                local_thresh = np.mean(filtered_block)
                binary_block = (filtered_block > local_thresh).astype(np.uint8) * 255
                binary[i:i+block_size, j:j+block_size] = binary_block
        
        return binary
    
    def create_oriented_gaussian(self, size, theta, sigma=1.0):
        """Create oriented Gaussian filter"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                x_rot = x * math.cos(theta) + y * math.sin(theta)
                y_rot = -x * math.sin(theta) + y * math.cos(theta)
                kernel[i, j] = math.exp(-(x_rot**2 + (2*y_rot)**2) / (2 * sigma**2))
        
        return kernel / np.sum(kernel)
    
    def thin_ridges(self, binary_image):
        """Zhang-Suen thinning algorithm"""
        skeleton = binary_image.copy() // 255
        changed = True
        
        while changed:
            changed = False
            markers = np.zeros_like(skeleton)
            for i in range(1, skeleton.shape[0]-1):
                for j in range(1, skeleton.shape[1]-1):
                    if skeleton[i, j] == 1:
                        P = [skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1],
                             skeleton[i+1, j+1], skeleton[i+1, j], skeleton[i+1, j-1],
                             skeleton[i, j-1], skeleton[i-1, j-1], skeleton[i-1, j]]
                        BP = sum(P[:-1])
                        if 2 <= BP <= 6:
                            transitions = sum((P[k] == 0 and P[k+1] == 1) for k in range(8))
                            if transitions == 1:
                                if P[0] * P[2] * P[4] == 0 and P[2] * P[4] * P[6] == 0:
                                    markers[i, j] = 1
            
            skeleton = skeleton - markers
            changed = np.any(markers)
            
            markers = np.zeros_like(skeleton)
            for i in range(1, skeleton.shape[0]-1):
                for j in range(1, skeleton.shape[1]-1):
                    if skeleton[i, j] == 1:
                        P = [skeleton[i-1, j], skeleton[i-1, j+1], skeleton[i, j+1],
                             skeleton[i+1, j+1], skeleton[i+1, j], skeleton[i+1, j-1],
                             skeleton[i, j-1], skeleton[i-1, j-1], skeleton[i-1, j]]
                        BP = sum(P[:-1])
                        if 2 <= BP <= 6:
                            transitions = sum((P[k] == 0 and P[k+1] == 1) for k in range(8))
                            if transitions == 1:
                                if P[0] * P[2] * P[6] == 0 and P[0] * P[4] * P[6] == 0:
                                    markers[i, j] = 1
            
            skeleton = skeleton - markers
            changed = changed or np.any(markers)
        
        return skeleton * 255
    
    def detect_minutiae(self, skeleton, orientation_map):
        """Detect minutiae points from skeletonized image"""
        minutiae = []
        
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i, j] > 0:
                    neighborhood = skeleton[i-1:i+2, j-1:j+2] // 255
                    crossings = np.sum(neighborhood)
                    
                    if crossings == 2:  # Ridge ending
                        angle = orientation_map[i, j]
                        minutiae.append({'x': j, 'y': i, 'angle': angle, 'type': 'ending'})
                    elif crossings == 4:  # Bifurcation
                        angle = orientation_map[i, j]
                        minutiae.append({'x': j, 'y': i, 'angle': angle, 'type': 'bifurcation'})
        
        return minutiae
    
    def filter_minutiae(self, minutiae, skeleton, min_distance=10):
        """Filter minutiae to achieve target count"""
        if len(minutiae) <= self.target_minutiae_count:
            return minutiae
        
        quality_scores = []
        for minutia in minutiae:
            x, y = minutia['x'], minutia['y']
            quality = self.calculate_minutiae_quality(skeleton, x, y)
            quality_scores.append(quality)
        
        sorted_indices = np.argsort(quality_scores)[::-1]
        filtered_minutiae = [minutiae[i] for i in sorted_indices[:self.target_minutiae_count]]
        
        return filtered_minutiae
    
    def calculate_minutiae_quality(self, skeleton, x, y, window_size=5):
        """Calculate quality score for minutiae point"""
        h, w = skeleton.shape
        x1 = max(0, x - window_size)
        y1 = max(0, y - window_size)
        x2 = min(w, x + window_size + 1)
        y2 = min(h, y + window_size + 1)
        
        roi = skeleton[y1:y2, x1:x2] // 255
        if roi.size == 0:
            return 0
        
        ridge_density = np.sum(roi) / roi.size
        return ridge_density
    
    def extract_aligned_patch(self, image, x, y, angle, patch_size=96):
        """Extract patch aligned with minutiae orientation"""
        h, w = image.shape
        larger_size = int(math.sqrt(2) * patch_size)
        
        x = max(larger_size//2, min(w - larger_size//2, x))
        y = max(larger_size//2, min(h - larger_size//2, y))
        
        x1, y1 = max(0, x - larger_size//2), max(0, y - larger_size//2)
        x2, y2 = min(w, x + larger_size//2), min(h, y + larger_size//2)
        
        larger_patch = image[y1:y2, x1:x2]
        
        if larger_patch.size == 0:
            return None
        
        center = (larger_patch.shape[1]//2, larger_patch.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, math.degrees(angle), 1.0)
        rotated_patch = cv2.warpAffine(larger_patch, rotation_matrix, 
                                     (larger_patch.shape[1], larger_patch.shape[0]))
        
        center_x, center_y = rotated_patch.shape[1]//2, rotated_patch.shape[0]//2
        start_x = center_x - patch_size//2
        start_y = center_y - patch_size//2
        final_patch = rotated_patch[start_y:start_y+patch_size, start_x:start_x+patch_size]
        
        return final_patch

def extract_minutiae_patches():
    """Main function to extract minutiae-based patches"""
    base_path = r"C:\Users\Mohamed\livdet2015_crossmatch"
    output_base = r"C:\Users\Mohamed\livdet2015_patches"
    extractor = CaoMinutiaeExtractor(target_minutiae_count=46)
    
    splits = {
        'Training': {
            'Live': ['Live'],
            'Fake': ['Ecoflex', 'Playdoh']
        },
        'Testing': {
            'Live': ['Live'],
            'Fake': ['Body Double', 'Ecoflex', 'Gelatin', 'OOMOO', 'Playdoh']
        }
    }
    
    metadata = []
    
    for split, categories in splits.items():
        for authenticity, materials in categories.items():
            for material in materials:
                if authenticity == 'Live':
                    input_dir = os.path.join(base_path, split, 'Live')
                    output_class = 'live'
                else:
                    input_dir = os.path.join(base_path, split, 'Fake', material)
                    output_class = 'spoof'
                
                print(f"Processing: {input_dir}")
                
                if not os.path.exists(input_dir):
                    print(f"Directory not found: {input_dir}")
                    continue
                
                # Create output directory for patches
                patch_output_dir = os.path.join(output_base, split, output_class, material)
                os.makedirs(patch_output_dir, exist_ok=True)
                
                for filename in os.listdir(input_dir):
                    if filename.lower().endswith('.bmp'):
                        image_path = os.path.join(input_dir, filename)
                        
                        try:
                            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                            if image is None:
                                continue
                            
                            # Process image
                            standardized = extractor.standardize_to_500dpi(image, 1000)
                            enhanced = extractor.enhance_image(standardized)
                            orientation_map = extractor.compute_orientation_map(enhanced)
                            binary = extractor.binarize_image(enhanced, orientation_map)
                            skeleton = extractor.thin_ridges(binary)
                            minutiae = extractor.detect_minutiae(skeleton, orientation_map)
                            filtered_minutiae = extractor.filter_minutiae(minutiae, skeleton)
                            
                            # Extract and save patches
                            for idx, minutia in enumerate(filtered_minutiae):
                                patch = extractor.extract_aligned_patch(
                                    enhanced, minutia['x'], minutia['y'], minutia['angle'], 96)
                                
                                if patch is not None:
                                    # Resize to 224x224 for MobileNet
                                    patch_resized = cv2.resize(patch, (224, 224))
                                    
                                    patch_filename = f"{os.path.splitext(filename)[0]}_minutiae_{idx}.png"
                                    patch_path = os.path.join(patch_output_dir, patch_filename)
                                    cv2.imwrite(patch_path, patch_resized)
                                    
                                    metadata.append({
                                        'patch_path': patch_path,
                                        'original_image': filename,
                                        'split': split,
                                        'class': output_class,
                                        'material': material,
                                        'minutiae_x': minutia['x'],
                                        'minutiae_y': minutia['y'],
                                        'minutiae_angle': minutia['angle'],
                                        'minutiae_type': minutia['type']
                                    })
                            
                            print(f"  {filename}: {len(filtered_minutiae)} patches")
                            
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_base, "patch_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPatch extraction complete! Metadata saved to: {metadata_path}")
    return metadata

if __name__ == "__main__":
    metadata = extract_minutiae_patches()
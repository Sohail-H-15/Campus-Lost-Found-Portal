"""
Image Matching Module using ResNet and SIFT
Provides image similarity comparison for the Lost & Found portal
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import base64

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImageMatcher:
    """Image matching using ResNet (deep learning) and SIFT (feature matching)"""
    
    def __init__(self, method: str = 'resnet'):
        """
        Initialize the image matcher
        
        Args:
            method: 'resnet' for deep learning or 'sift' for feature matching
        """
        self.method = method.lower()
        
        if self.method == 'resnet':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch and torchvision required for ResNet. Install: pip install torch torchvision pillow")
            self._init_resnet()
        elif self.method == 'sift':
            if not CV2_AVAILABLE:
                raise ImportError("OpenCV required for SIFT. Install: pip install opencv-python")
            self._init_sift()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'resnet' or 'sift'")
    
    def _init_resnet(self):
        """Initialize ResNet50 model for feature extraction"""
        # Load pre-trained ResNet50 model
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final classification layer to get feature vectors
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _init_sift(self):
        """Initialize SIFT detector"""
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def extract_features_resnet(self, image_path: Path) -> np.ndarray:
        """
        Extract feature vector from image using ResNet50
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized feature vector as numpy array
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Flatten and normalize
            features = features.flatten()
            features = features / (np.linalg.norm(features) + 1e-8)  # L2 normalization
            
            return features
        except Exception as e:
            raise ValueError(f"Error extracting ResNet features from {image_path}: {str(e)}")
    
    def extract_features_sift(self, image_path: Path) -> Optional[List]:
        """
        Extract SIFT keypoints and descriptors from image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List containing keypoints and descriptors, or None if extraction fails
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            
            if descriptors is None:
                return None
            
            return [keypoints, descriptors]
        except Exception as e:
            raise ValueError(f"Error extracting SIFT features from {image_path}: {str(e)}")
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract features from image using the configured method
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector (ResNet) or serialized SIFT descriptors (SIFT)
        """
        if self.method == 'resnet':
            return self.extract_features_resnet(image_path)
        else:  # sift
            features = self.extract_features_sift(image_path)
            if features is None:
                return np.array([])
            # Serialize SIFT descriptors for storage
            _, descriptors = features
            return descriptors
    
    def compare_features_resnet(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compare two ResNet feature vectors using cosine similarity
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        # Cosine similarity
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
        # Normalize to [0, 1] range (cosine similarity is [-1, 1])
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def compare_features_sift(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compare two SIFT descriptor sets
        
        Args:
            feat1: First set of SIFT descriptors
            feat2: Second set of SIFT descriptors
            
        Returns:
            Similarity score between 0 and 1
        """
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        
        try:
            # Ensure descriptors are numpy arrays with proper shape
            if not isinstance(feat1, np.ndarray):
                feat1 = np.array(feat1)
            if not isinstance(feat2, np.ndarray):
                feat2 = np.array(feat2)
            
            # Check if descriptors are valid
            if len(feat1.shape) != 2 or len(feat2.shape) != 2:
                return 0.0
            
            # Need at least 2 descriptors for knnMatch
            if len(feat1) < 2 or len(feat2) < 2:
                # Fall back to brute force matching for small sets
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(feat1, feat2)
                good_matches = matches
            else:
                # Use FLANN matcher for better performance
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(feat1, feat2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
            
            # Calculate similarity based on number of good matches
            max_possible_matches = min(len(feat1), len(feat2))
            if max_possible_matches == 0:
                return 0.0
            
            similarity = len(good_matches) / max_possible_matches
            return min(similarity, 1.0)
        except Exception as e:
            print(f"Error comparing SIFT features: {str(e)}")
            return 0.0
    
    def compare_features(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compare two feature vectors using the configured method
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.method == 'resnet':
            return self.compare_features_resnet(feat1, feat2)
        else:  # sift
            return self.compare_features_sift(feat1, feat2)
    
    def serialize_features(self, features: np.ndarray) -> bytes:
        """
        Serialize feature vector for database storage
        
        Args:
            features: Feature vector as numpy array
            
        Returns:
            Serialized bytes
        """
        if self.method == 'resnet':
            # Store as base64-encoded pickle for ResNet (smaller vectors)
            return base64.b64encode(pickle.dumps(features.astype(np.float32)))
        else:  # sift
            # SIFT descriptors are already numpy arrays
            return base64.b64encode(pickle.dumps(features))
    
    def deserialize_features(self, serialized: bytes) -> np.ndarray:
        """
        Deserialize feature vector from database
        
        Args:
            serialized: Serialized bytes from database (or string/None)
            
        Returns:
            Feature vector as numpy array
        """
        if serialized is None:
            return np.array([])
        
        if isinstance(serialized, str):
            # If stored as string, decode first
            serialized = serialized.encode('utf-8')
        
        if isinstance(serialized, bytes):
            try:
                data = pickle.loads(base64.b64decode(serialized))
                return np.array(data)
            except Exception as e:
                raise ValueError(f"Failed to deserialize features: {e}")
        
        return np.array([])


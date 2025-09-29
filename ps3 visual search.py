import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from image_processor import ImageProcessor
from feature_matcher import FeatureMatcher
from satellite_searcher import SatelliteSearcher
from utils import save_results, validate_tiff_file, load_sample_images

# Initialize session state
if 'sample_chips' not in st.session_state:
    st.session_state.sample_chips = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'current_satellite_image' not in st.session_state:
    st.session_state.current_satellite_image = None

def main():
    st.title("🛰️ Satellite Imagery Visual Search & Detection System")
    st.markdown("Upload sample image chips or select regions from satellite imagery to find similar objects across multispectral datasets.")
   
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Upload Sample Chips", "Select from Satellite Image", "Configure Search", "Run Search", "View Results"]
    )
   
    if mode == "Upload Sample Chips":
        upload_sample_chips()
    elif mode == "Select from Satellite Image":
        select_from_satellite_image()
    elif mode == "Configure Search":
        configure_search()
    elif mode == "Run Search":
        run_search()
    elif mode == "View Results":
        view_results()

def upload_sample_chips():
    st.header("📁 Upload Sample Image Chips")
    st.markdown("Upload up to 5 sample image chips of the object/feature you want to search for.")
   
    uploaded_files = st.file_uploader(
        "Choose sample image files",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        accept_multiple_files=True,
        help="Upload 1-5 sample images of the object/feature you want to detect"
    )
   
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Please upload maximum 5 sample images.")
            return
       
        st.subheader("Uploaded Sample Chips")
        cols = st.columns(min(len(uploaded_files), 3))
       
        sample_chips = []
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
               
                # Load and process image
                image = Image.open(tmp_path)
                image_array = np.array(image)
               
                # Display in column
                with cols[idx % 3]:
                    st.image(image, caption=f"Sample {idx+1}: {uploaded_file.name}", use_column_width=True)
               
                sample_chips.append({
                    'name': uploaded_file.name,
                    'path': tmp_path,
                    'image': image_array
                })
               
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
       
        if sample_chips:
            st.session_state.sample_chips = sample_chips
            st.success(f"Successfully loaded {len(sample_chips)} sample chips!")
           
            # Extract features button
            if st.button("Extract Features from Sample Chips"):
                with st.spinner("Extracting features..."):
                    processor = ImageProcessor()
                    for chip in st.session_state.sample_chips:
                        features = processor.extract_features(chip['image'])
                        chip['features'] = features
                    st.success("Features extracted successfully!")

def select_from_satellite_image():
    st.header("🎯 Select Regions from Satellite Image")
    st.markdown("Load a satellite TIFF image and draw bounding boxes around objects of interest.")
   
    uploaded_tiff = st.file_uploader(
        "Upload Satellite TIFF Image",
        type=['tiff', 'tif'],
        help="Upload a multispectral TIFF file for region selection"
    )
   
    if uploaded_tiff:
        try:
            # Save uploaded TIFF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp_file:
                tmp_file.write(uploaded_tiff.getvalue())
                tiff_path = tmp_file.name
           
            # Validate TIFF file
            if not validate_tiff_file(tiff_path):
                st.error("Invalid TIFF file format. Please upload a valid multispectral TIFF file.")
                return
           
            # Load and display TIFF
            with rasterio.open(tiff_path) as src:
                # Read RGB bands for display (bands 3,2,1 for Red,Green,Blue)
                if src.count >= 3:
                    rgb_image = np.dstack([
                        src.read(3),  # Red
                        src.read(2),  # Green  
                        src.read(1)   # Blue
                    ])
                    # Normalize for display
                    rgb_image = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8)
                else:
                    st.error("TIFF file must have at least 3 bands.")
                    return
           
            st.session_state.current_satellite_image = {
                'path': tiff_path,
                'rgb_display': rgb_image,
                'name': uploaded_tiff.name
            }
           
            st.subheader("Draw Bounding Boxes")
            st.markdown("Draw rectangles around objects of interest in the satellite image.")
           
            # Create canvas for drawing
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.1)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=Image.fromarray(rgb_image),
                update_streamlit=True,
                height=min(rgb_image.shape[0], 600),
                width=min(rgb_image.shape[1], 800),
                drawing_mode="rect",
                key="satellite_canvas"
            )
           
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects and st.button("Extract Selected Regions"):
                    extracted_chips = []
                    processor = ImageProcessor()
                   
                    for idx, obj in enumerate(objects):
                        if obj["type"] == "rect":
                            # Get bounding box coordinates
                            left = int(obj["left"])
                            top = int(obj["top"])
                            width = int(obj["width"])
                            height = int(obj["height"])
                           
                            # Extract chip from original image
                            chip = rgb_image[top:top+height, left:left+width]
                           
                            if chip.size > 0:
                                extracted_chips.append({
                                    'name': f"selected_region_{idx+1}",
                                    'image': chip,
                                    'bbox': (left, top, left+width, top+height),
                                    'features': processor.extract_features(chip)
                                })
                   
                    if extracted_chips:
                        # Add to existing sample chips
                        st.session_state.sample_chips.extend(extracted_chips)
                        st.success(f"Extracted {len(extracted_chips)} regions successfully!")
                       
                        # Display extracted chips
                        st.subheader("Extracted Regions")
                        cols = st.columns(min(len(extracted_chips), 3))
                        for idx, chip in enumerate(extracted_chips):
                            with cols[idx % 3]:
                                st.image(chip['image'], caption=chip['name'], use_column_width=True)
                   
        except Exception as e:
            st.error(f"Error processing TIFF file: {str(e)}")

def configure_search():
    st.header("⚙️ Configure Search Parameters")
   
    if not st.session_state.sample_chips:
        st.warning("Please upload sample chips or select regions first.")
        return
   
    st.subheader("Current Sample Chips")
    cols = st.columns(min(len(st.session_state.sample_chips), 3))
    for idx, chip in enumerate(st.session_state.sample_chips):
        with cols[idx % 3]:
            st.image(chip['image'], caption=chip['name'], use_column_width=True)
   
    st.subheader("Search Configuration")
   
    # Target directory selection
    target_dir = st.text_input(
        "Target Directory Path",
        placeholder="/path/to/satellite/images",
        help="Path to directory containing satellite TIFF images to search"
    )
   
    # Output directory selection
    output_dir = st.text_input(
        "Output Directory Path",
        placeholder="/path/to/output",
        help="Path to directory where results will be saved"
    )
   
    # Search parameters
    st.subheader("Search Parameters")
   
    col1, col2 = st.columns(2)
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for matches"
        )
       
        max_matches_per_image = st.number_input(
            "Max Matches per Image",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of matches to find per satellite image"
        )
   
    with col2:
        feature_detector = st.selectbox(
            "Feature Detector",
            ["SIFT", "ORB", "AKAZE"],
            help="Algorithm for feature detection and extraction"
        )
       
        search_window_size = st.number_input(
            "Search Window Size",
            min_value=32,
            max_value=512,
            value=128,
            step=32,
            help="Size of sliding window for object detection"
        )
   
    # Store configuration
    if st.button("Save Configuration"):
        if not target_dir or not output_dir:
            st.error("Please specify both target and output directories.")
            return
       
        if not os.path.exists(target_dir):
            st.error(f"Target directory does not exist: {target_dir}")
            return
       
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
       
        st.session_state.search_config = {
            'target_dir': target_dir,
            'output_dir': output_dir,
            'similarity_threshold': similarity_threshold,
            'max_matches_per_image': max_matches_per_image,
            'feature_detector': feature_detector,
            'search_window_size': search_window_size
        }
       
        st.success("Configuration saved successfully!")

def run_search():
    st.header("🔍 Run Visual Search")
   
    if not st.session_state.sample_chips:
        st.warning("Please upload sample chips first.")
        return
   
    if 'search_config' not in st.session_state:
        st.warning("Please configure search parameters first.")
        return
   
    config = st.session_state.search_config
   
    st.subheader("Search Configuration Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Target Directory:** {config['target_dir']}")
        st.write(f"**Output Directory:** {config['output_dir']}")
        st.write(f"**Sample Chips:** {len(st.session_state.sample_chips)}")
    with col2:
        st.write(f"**Similarity Threshold:** {config['similarity_threshold']}")
        st.write(f"**Feature Detector:** {config['feature_detector']}")
        st.write(f"**Max Matches per Image:** {config['max_matches_per_image']}")
   
    if st.button("Start Search", type="primary"):
        with st.spinner("Searching satellite images..."):
            try:
                # Initialize searcher
                searcher = SatelliteSearcher(
                    feature_detector=config['feature_detector'],
                    similarity_threshold=config['similarity_threshold'],
                    max_matches=config['max_matches_per_image'],
                    window_size=config['search_window_size']
                )
               
                # Get list of TIFF files in target directory
                tiff_files = []
                for file in os.listdir(config['target_dir']):
                    if file.lower().endswith(('.tiff', '.tif')):
                        full_path = os.path.join(config['target_dir'], file)
                        if validate_tiff_file(full_path):
                            tiff_files.append(full_path)
               
                if not tiff_files:
                    st.error("No valid TIFF files found in target directory.")
                    return
               
                st.info(f"Found {len(tiff_files)} TIFF files to search.")
               
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
               
                all_results = []
               
                # Search each TIFF file
                for idx, tiff_path in enumerate(tiff_files):
                    status_text.text(f"Searching {os.path.basename(tiff_path)}...")
                   
                    # Search for each sample chip in this TIFF
                    for chip in st.session_state.sample_chips:
                        if 'features' not in chip:
                            processor = ImageProcessor()
                            chip['features'] = processor.extract_features(chip['image'])
                       
                        matches = searcher.search_in_satellite_image(
                            tiff_path,
                            chip['features'],
                            chip['name']
                        )
                        all_results.extend(matches)
                   
                    # Update progress
                    progress = (idx + 1) / len(tiff_files)
                    progress_bar.progress(progress)
               
                status_text.text("Search completed!")
               
                # Store results
                st.session_state.search_results = all_results
               
                # Save results to file
                output_file = os.path.join(config['output_dir'], 'search_results.txt')
                save_results(all_results, output_file)
               
                st.success(f"Search completed! Found {len(all_results)} matches.")
                st.info(f"Results saved to: {output_file}")
               
                # Display summary
                if all_results:
                    df = pd.DataFrame(all_results)
                    st.subheader("Results Summary")
                    st.write(f"Total matches found: {len(all_results)}")
                    st.write(f"Average similarity score: {df['similarity_score'].mean():.3f}")
                    st.write(f"Images with matches: {df['target_imagery_file_name'].nunique()}")
               
            except Exception as e:
                st.error(f"Error during search: {str(e)}")

def view_results():
    st.header("📊 View Search Results")
   
    if not st.session_state.search_results:
        st.warning("No search results available. Please run a search first.")
        return
   
    results = st.session_state.search_results
    df = pd.DataFrame(results)
   
    # Results summary
    st.subheader("Results Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(results))
    with col2:
        st.metric("Avg Similarity", f"{df['similarity_score'].mean():.3f}")
    with col3:
        st.metric("Images Searched", df['target_imagery_file_name'].nunique())
    with col4:
        st.metric("Object Types", df['searched_object_name'].nunique())
   
    # Filter controls
    st.subheader("Filter Results")
    col1, col2 = st.columns(2)
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    with col2:
        selected_objects = st.multiselect(
            "Object Types",
            options=df['searched_object_name'].unique(),
            default=df['searched_object_name'].unique()
        )
   
    # Apply filters
    filtered_df = df[
        (df['similarity_score'] >= min_similarity) &
        (df['searched_object_name'].isin(selected_objects))
    ]
   
    st.subheader("Filtered Results")
    st.write(f"Showing {len(filtered_df)} of {len(df)} results")
   
    # Display results table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            'similarity_score': st.column_config.NumberColumn(
                'Similarity Score',
                format="%.3f"
            )
        }
    )
   
    # Download results
    if st.button("Download Filtered Results"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_search_results.csv",
            mime="text/csv"
        )
   
    # Visualization
    if len(filtered_df) > 0:
        st.subheader("Results Visualization")
       
        # Similarity score distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
       
        ax1.hist(filtered_df['similarity_score'], bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Similarity Scores')
       
        # Matches per image
        matches_per_image = filtered_df['target_imagery_file_name'].value_counts().head(10)
        ax2.bar(range(len(matches_per_image)), matches_per_image.values)
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Number of Matches')
        ax2.set_title('Top 10 Images by Match Count')
       
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
    import cv2
import numpy as np
from skimage import feature, filters, segmentation
from sklearn.preprocessing import StandardScaler
import rasterio
from PIL import Image

class ImageProcessor:
    """Handles image processing operations for satellite imagery and sample chips."""
   
    def __init__(self, feature_detector='SIFT'):
        """Initialize the image processor with specified feature detector."""
        self.feature_detector_type = feature_detector
        self.feature_detector = self._get_feature_detector(feature_detector)
        self.scaler = StandardScaler()
   
    def _get_feature_detector(self, detector_type):
        """Get the appropriate feature detector based on type."""
        if detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=500)
        elif detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=500)
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported feature detector: {detector_type}")
   
    def preprocess_satellite_image(self, image_path):
        """Load and preprocess satellite TIFF image."""
        try:
            with rasterio.open(image_path) as src:
                # Read all bands
                bands = []
                for i in range(1, src.count + 1):
                    band = src.read(i)
                    bands.append(band)
               
                # Stack bands
                image = np.dstack(bands)
               
                # Handle different band configurations
                if src.count >= 4:
                    # Multispectral: Blue, Green, Red, NIR
                    rgb_image = image[:, :, [2, 1, 0]]  # Red, Green, Blue
                    nir_image = image[:, :, 3]  # NIR
                elif src.count >= 3:
                    # RGB only
                    rgb_image = image[:, :, [2, 1, 0]]  # Red, Green, Blue
                    nir_image = None
                else:
                    raise ValueError("Image must have at least 3 bands")
               
                # Normalize to 0-255 range
                rgb_image = self._normalize_image(rgb_image)
               
                return {
                    'rgb': rgb_image,
                    'nir': nir_image,
                    'all_bands': image,
                    'metadata': {
                        'width': src.width,
                        'height': src.height,
                        'bands': src.count,
                        'crs': str(src.crs) if src.crs else None,
                        'transform': src.transform
                    }
                }
               
        except Exception as e:
            raise Exception(f"Error processing satellite image {image_path}: {str(e)}")
   
    def _normalize_image(self, image):
        """Normalize image to 0-255 range."""
        if image.dtype != np.uint8:
            # Normalize each channel separately
            normalized = np.zeros_like(image, dtype=np.uint8)
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                channel_min, channel_max = np.percentile(channel, [2, 98])
                normalized[:, :, i] = np.clip(
                    ((channel - channel_min) / (channel_max - channel_min) * 255),
                    0, 255
                ).astype(np.uint8)
            return normalized
        return image
   
    def extract_features(self, image):
        """Extract comprehensive features from an image chip."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
           
            # Ensure proper data type
            if gray.dtype != np.uint8:
                gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
           
            features = {}
           
            # 1. Keypoint-based features
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            features['keypoints'] = keypoints
            features['descriptors'] = descriptors
           
            # 2. Texture features
            features['texture'] = self._extract_texture_features(gray)
           
            # 3. Shape features
            features['shape'] = self._extract_shape_features(gray)
           
            # 4. Color features (if RGB image)
            if len(image.shape) == 3:
                features['color'] = self._extract_color_features(image)
           
            # 5. Statistical features
            features['statistics'] = self._extract_statistical_features(gray)
           
            return features
           
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
   
    def _extract_texture_features(self, gray_image):
        """Extract texture features using Local Binary Pattern and GLCM."""
        texture_features = {}
       
        try:
            # Local Binary Pattern
            lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
            texture_features['lbp_histogram'] = lbp_hist
           
            # Edge density
            edges = feature.canny(gray_image, sigma=1.0)
            texture_features['edge_density'] = np.sum(edges) / edges.size
           
        except Exception as e:
            print(f"Error extracting texture features: {str(e)}")
            texture_features = {'lbp_histogram': np.zeros(10), 'edge_density': 0.0}
       
        return texture_features
   
    def _extract_shape_features(self, gray_image):
        """Extract shape-based features."""
        shape_features = {}
       
        try:
            # Find contours
            contours, _ = cv2.findContours(
                cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
           
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
               
                # Contour area
                shape_features['area'] = cv2.contourArea(largest_contour)
               
                # Contour perimeter
                shape_features['perimeter'] = cv2.arcLength(largest_contour, True)
               
                # Aspect ratio of bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                shape_features['aspect_ratio'] = float(w) / h if h > 0 else 0
               
                # Extent (ratio of contour area to bounding rectangle area)
                rect_area = w * h
                shape_features['extent'] = shape_features['area'] / rect_area if rect_area > 0 else 0
               
                # Solidity (ratio of contour area to convex hull area)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                shape_features['solidity'] = shape_features['area'] / hull_area if hull_area > 0 else 0
               
            else:
                shape_features = {
                    'area': 0, 'perimeter': 0, 'aspect_ratio': 1,
                    'extent': 0, 'solidity': 0
                }
               
        except Exception as e:
            print(f"Error extracting shape features: {str(e)}")
            shape_features = {
                'area': 0, 'perimeter': 0, 'aspect_ratio': 1,
                'extent': 0, 'solidity': 0
            }
       
        return shape_features
   
    def _extract_color_features(self, rgb_image):
        """Extract color-based features."""
        color_features = {}
       
        try:
            # Mean RGB values
            color_features['mean_rgb'] = np.mean(rgb_image.reshape(-1, 3), axis=0)
           
            # Standard deviation of RGB values
            color_features['std_rgb'] = np.std(rgb_image.reshape(-1, 3), axis=0)
           
            # Color histograms
            color_features['hist_r'] = np.histogram(rgb_image[:, :, 0], bins=16, range=(0, 256))[0]
            color_features['hist_g'] = np.histogram(rgb_image[:, :, 1], bins=16, range=(0, 256))[0]
            color_features['hist_b'] = np.histogram(rgb_image[:, :, 2], bins=16, range=(0, 256))[0]
           
            # Normalize histograms
            total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
            color_features['hist_r'] = color_features['hist_r'] / total_pixels
            color_features['hist_g'] = color_features['hist_g'] / total_pixels
            color_features['hist_b'] = color_features['hist_b'] / total_pixels
           
        except Exception as e:
            print(f"Error extracting color features: {str(e)}")
            color_features = {
                'mean_rgb': np.zeros(3),
                'std_rgb': np.zeros(3),
                'hist_r': np.zeros(16),
                'hist_g': np.zeros(16),
                'hist_b': np.zeros(16)
            }
       
        return color_features
   
    def _extract_statistical_features(self, gray_image):
        """Extract statistical features from the image."""
        stats = {}
       
        try:
            stats['mean'] = np.mean(gray_image)
            stats['std'] = np.std(gray_image)
            stats['variance'] = np.var(gray_image)
            stats['skewness'] = self._calculate_skewness(gray_image)
            stats['kurtosis'] = self._calculate_kurtosis(gray_image)
            stats['entropy'] = self._calculate_entropy(gray_image)
           
        except Exception as e:
            print(f"Error extracting statistical features: {str(e)}")
            stats = {
                'mean': 0, 'std': 0, 'variance': 0,
                'skewness': 0, 'kurtosis': 0, 'entropy': 0
            }
       
        return stats
   
    def _calculate_skewness(self, image):
        """Calculate skewness of image intensity distribution."""
        flat = image.flatten()
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        if std_val == 0:
            return 0
        return np.mean(((flat - mean_val) / std_val) ** 3)
   
    def _calculate_kurtosis(self, image):
        """Calculate kurtosis of image intensity distribution."""
        flat = image.flatten()
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        if std_val == 0:
            return 0
        return np.mean(((flat - mean_val) / std_val) ** 4) - 3
   
    def _calculate_entropy(self, image):
        """Calculate entropy of image."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zero entries
        return -np.sum(hist * np.log2(hist))
   
    def create_sliding_windows(self, image, window_size, stride=None):
        """Create sliding windows across the image for object detection."""
        if stride is None:
            stride = window_size // 4  # 75% overlap by default
       
        height, width = image.shape[:2]
        windows = []
       
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                window = image[y:y+window_size, x:x+window_size]
                windows.append({
                    'image': window,
                    'bbox': (x, y, x+window_size, y+window_size)
                })
       
        return windows
        import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN

class FeatureMatcher:
    """Handles feature matching and similarity computation between image chips."""
   
    def __init__(self, matcher_type='FLANN', distance_threshold=0.7):
        """Initialize the feature matcher."""
        self.matcher_type = matcher_type
        self.distance_threshold = distance_threshold
        self.matcher = self._get_matcher(matcher_type)
        self.scaler = StandardScaler()
   
    def _get_matcher(self, matcher_type):
        """Get the appropriate feature matcher."""
        if matcher_type == 'FLANN':
            # FLANN parameters for SIFT/SURF
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == 'BF':
            # Brute force matcher
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise ValueError(f"Unsupported matcher type: {matcher_type}")
   
    def compute_similarity(self, features1, features2):
        """Compute comprehensive similarity between two feature sets."""
        if features1 is None or features2 is None:
            return 0.0
       
        try:
            similarities = {}
           
            # 1. Keypoint descriptor matching
            desc_similarity = self._compute_descriptor_similarity(
                features1.get('descriptors'),
                features2.get('descriptors')
            )
            similarities['descriptors'] = desc_similarity
           
            # 2. Texture similarity
            texture_similarity = self._compute_texture_similarity(
                features1.get('texture'),
                features2.get('texture')
            )
            similarities['texture'] = texture_similarity
           
            # 3. Shape similarity
            shape_similarity = self._compute_shape_similarity(
                features1.get('shape'),
                features2.get('shape')
            )
            similarities['shape'] = shape_similarity
           
            # 4. Color similarity (if available)
            color_similarity = 0.0
            if 'color' in features1 and 'color' in features2:
                color_similarity = self._compute_color_similarity(
                    features1['color'],
                    features2['color']
                )
            similarities['color'] = color_similarity
           
            # 5. Statistical similarity
            stats_similarity = self._compute_statistical_similarity(
                features1.get('statistics'),
                features2.get('statistics')
            )
            similarities['statistics'] = stats_similarity
           
            # Compute weighted overall similarity
            weights = {
                'descriptors': 0.4,
                'texture': 0.2,
                'shape': 0.15,
                'color': 0.15,
                'statistics': 0.1
            }
           
            overall_similarity = sum(
                similarities[key] * weights[key]
                for key in weights.keys()
            )
           
            return max(0.0, min(1.0, overall_similarity))
           
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0
   
    def _compute_descriptor_similarity(self, desc1, desc2):
        """Compute similarity based on keypoint descriptors."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
       
        try:
            # Convert to float32 if needed
            if desc1.dtype != np.float32:
                desc1 = desc1.astype(np.float32)
            if desc2.dtype != np.float32:
                desc2 = desc2.astype(np.float32)
           
            # Find matches
            if self.matcher_type == 'FLANN' and len(desc1) >= 2 and len(desc2) >= 2:
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
               
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.distance_threshold * n.distance:
                            good_matches.append(m)
               
                # Calculate similarity based on number of good matches
                max_matches = min(len(desc1), len(desc2))
                if max_matches > 0:
                    return len(good_matches) / max_matches
                else:
                    return 0.0
            else:
                # Use brute force for small descriptor sets
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(desc1, desc2)
               
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
               
                # Calculate similarity based on average distance of best matches
                if matches:
                    # Take top 20% of matches or at least 5
                    num_good = max(5, len(matches) // 5)
                    good_matches = matches[:num_good]
                    avg_distance = np.mean([m.distance for m in good_matches])
                   
                    # Normalize distance to similarity (lower distance = higher similarity)
                    # Assuming maximum reasonable distance is 256 for SIFT descriptors
                    max_distance = 256.0
                    similarity = max(0.0, 1.0 - (avg_distance / max_distance))
                    return similarity
                else:
                    return 0.0
                   
        except Exception as e:
            print(f"Error computing descriptor similarity: {str(e)}")
            return 0.0
   
    def _compute_texture_similarity(self, texture1, texture2):
        """Compute similarity based on texture features."""
        if texture1 is None or texture2 is None:
            return 0.0
       
        try:
            similarities = []
           
            # LBP histogram similarity
            if 'lbp_histogram' in texture1 and 'lbp_histogram' in texture2:
                lbp_sim = 1.0 - np.sum(np.abs(texture1['lbp_histogram'] - texture2['lbp_histogram'])) / 2.0
                similarities.append(lbp_sim)
           
            # Edge density similarity
            if 'edge_density' in texture1 and 'edge_density' in texture2:
                edge_diff = abs(texture1['edge_density'] - texture2['edge_density'])
                edge_sim = max(0.0, 1.0 - edge_diff)
                similarities.append(edge_sim)
           
            return np.mean(similarities) if similarities else 0.0
           
        except Exception as e:
            print(f"Error computing texture similarity: {str(e)}")
            return 0.0
   
    def _compute_shape_similarity(self, shape1, shape2):
        """Compute similarity based on shape features."""
        if shape1 is None or shape2 is None:
            return 0.0
       
        try:
            similarities = []
            shape_features = ['aspect_ratio', 'extent', 'solidity']
           
            for feature in shape_features:
                if feature in shape1 and feature in shape2:
                    val1, val2 = shape1[feature], shape2[feature]
                    if val1 == 0 and val2 == 0:
                        sim = 1.0
                    elif val1 == 0 or val2 == 0:
                        sim = 0.0
                    else:
                        # Compute relative difference
                        diff = abs(val1 - val2) / max(val1, val2)
                        sim = max(0.0, 1.0 - diff)
                    similarities.append(sim)
           
            return np.mean(similarities) if similarities else 0.0
           
        except Exception as e:
            print(f"Error computing shape similarity: {str(e)}")
            return 0.0
   
    def _compute_color_similarity(self, color1, color2):
        """Compute similarity based on color features."""
        if color1 is None or color2 is None:
            return 0.0
       
        try:
            similarities = []
           
            # Mean RGB similarity
            if 'mean_rgb' in color1 and 'mean_rgb' in color2:
                rgb_diff = np.linalg.norm(color1['mean_rgb'] - color2['mean_rgb'])
                rgb_sim = max(0.0, 1.0 - rgb_diff / (255.0 * np.sqrt(3)))
                similarities.append(rgb_sim)
           
            # Color histogram similarities
            hist_features = ['hist_r', 'hist_g', 'hist_b']
            for hist_feature in hist_features:
                if hist_feature in color1 and hist_feature in color2:
                    # Use chi-square distance for histogram comparison
                    hist1, hist2 = color1[hist_feature], color2[hist_feature]
                    chi_sq = cv2.compareHist(
                        hist1.astype(np.float32),
                        hist2.astype(np.float32),
                        cv2.HISTCMP_CHISQR
                    )
                    # Convert chi-square to similarity (lower is better)
                    hist_sim = max(0.0, 1.0 / (1.0 + chi_sq))
                    similarities.append(hist_sim)
           
            return np.mean(similarities) if similarities else 0.0
           
        except Exception as e:
            print(f"Error computing color similarity: {str(e)}")
            return 0.0
   
    def _compute_statistical_similarity(self, stats1, stats2):
        """Compute similarity based on statistical features."""
        if stats1 is None or stats2 is None:
            return 0.0
       
        try:
            similarities = []
            stat_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'entropy']
           
            for feature in stat_features:
                if feature in stats1 and feature in stats2:
                    val1, val2 = stats1[feature], stats2[feature]
                   
                    if feature in ['mean', 'std', 'variance']:
                        # Normalize by image intensity range (0-255)
                        max_val = 255.0 if feature == 'mean' else 255.0**2
                        diff = abs(val1 - val2) / max_val
                    elif feature == 'entropy':
                        # Entropy ranges from 0 to log2(256) ≈ 8
                        diff = abs(val1 - val2) / 8.0
                    else:
                        # For skewness and kurtosis, use relative difference
                        if abs(val1) + abs(val2) > 0:
                            diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                        else:
                            diff = 0.0
                   
                    sim = max(0.0, 1.0 - diff)
                    similarities.append(sim)
           
            return np.mean(similarities) if similarities else 0.0
           
        except Exception as e:
            print(f"Error computing statistical similarity: {str(e)}")
            return 0.0
   
    def find_best_matches(self, query_features, candidate_features_list, top_k=10):
        """Find the best matching candidates for a query."""
        similarities = []
       
        for i, candidate_features in enumerate(candidate_features_list):
            similarity = self.compute_similarity(query_features, candidate_features)
            similarities.append((i, similarity))
       
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
       
        # Return top k matches
        return similarities[:top_k]
   
    def cluster_similar_detections(self, detections, similarity_threshold=0.8):
        """Cluster similar detections to reduce duplicates."""
        if len(detections) <= 1:
            return detections
       
        try:
            # Extract bounding boxes for clustering
            boxes = np.array([[d['x_min'], d['y_min'], d['x_max'], d['y_max']] for d in detections])
           
            # Compute pairwise IoU (Intersection over Union)
            ious = self._compute_pairwise_iou(boxes)
           
            # Use DBSCAN clustering based on IoU similarity
            # Convert IoU to distance (1 - IoU)
            distances = 1.0 - ious
           
            # Apply DBSCAN
            clustering = DBSCAN(
                eps=1.0 - similarity_threshold,
                min_samples=1,
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(distances)
           
            # Group detections by cluster and keep the best one from each cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(detections[i])
           
            # Select best detection from each cluster (highest similarity score)
            best_detections = []
            for cluster_detections in clusters.values():
                best_detection = max(cluster_detections, key=lambda x: x['similarity_score'])
                best_detections.append(best_detection)
           
            return best_detections
           
        except Exception as e:
            print(f"Error clustering detections: {str(e)}")
            return detections
   
    def _compute_pairwise_iou(self, boxes):
        """Compute pairwise IoU matrix for bounding boxes."""
        n = len(boxes)
        iou_matrix = np.zeros((n, n))
       
        for i in range(n):
            for j in range(n):
                if i == j:
                    iou_matrix[i, j] = 1.0
                else:
                    iou_matrix[i, j] = self._compute_iou(boxes[i], boxes[j])
       
        return iou_matrix
   
    def _compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
       
        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
       
        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)
       
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
       
        # Calculate IoU
        if union == 0:
            return 0.0
       
        return intersection / union
import os
import cv2
import numpy as np
from image_processor import ImageProcessor
from feature_matcher import FeatureMatcher
import rasterio
from concurrent.futures import ThreadPoolExecutor
import threading

class SatelliteSearcher:
    """Main class for searching objects in satellite imagery."""
   
    def __init__(self, feature_detector='SIFT', similarity_threshold=0.7,
                 max_matches=10, window_size=128, stride=None):
        """Initialize the satellite searcher."""
        self.feature_detector = feature_detector
        self.similarity_threshold = similarity_threshold
        self.max_matches = max_matches
        self.window_size = window_size
        self.stride = stride or window_size // 4
       
        self.image_processor = ImageProcessor(feature_detector)
        self.feature_matcher = FeatureMatcher()
       
        # Thread lock for thread-safe operations
        self.lock = threading.Lock()
   
    def search_in_satellite_image(self, satellite_image_path, query_features, object_name):
        """Search for objects in a single satellite image."""
        try:
            # Load and preprocess satellite image
            satellite_data = self.image_processor.preprocess_satellite_image(satellite_image_path)
            rgb_image = satellite_data['rgb']
           
            # Create sliding windows
            windows = self.image_processor.create_sliding_windows(
                rgb_image, self.window_size, self.stride
            )
           
            print(f"Processing {len(windows)} windows in {os.path.basename(satellite_image_path)}")
           
            # Search in each window
            detections = []
            batch_size = 100  # Process windows in batches to manage memory
           
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i+batch_size]
                batch_detections = self._process_window_batch(
                    batch_windows, query_features, object_name, satellite_image_path
                )
                detections.extend(batch_detections)
               
                # Update progress
                progress = min(100, (i + batch_size) * 100 // len(windows))
                if i % (batch_size * 5) == 0:  # Print progress every 5 batches
                    print(f"Progress: {progress}% ({len(detections)} matches found so far)")
           
            # Filter by similarity threshold
            filtered_detections = [
                d for d in detections
                if d['similarity_score'] >= self.similarity_threshold
            ]
           
            # Sort by similarity score (descending)
            filtered_detections.sort(key=lambda x: x['similarity_score'], reverse=True)
           
            # Apply non-maximum suppression to reduce overlapping detections
            final_detections = self.feature_matcher.cluster_similar_detections(
                filtered_detections, similarity_threshold=0.5
            )
           
            # Limit to max_matches
            final_detections = final_detections[:self.max_matches]
           
            print(f"Found {len(final_detections)} final matches in {os.path.basename(satellite_image_path)}")
           
            return final_detections
           
        except Exception as e:
            print(f"Error searching in {satellite_image_path}: {str(e)}")
            return []
   
    def _process_window_batch(self, windows, query_features, object_name, satellite_image_path):
        """Process a batch of windows for object detection."""
        detections = []
       
        for window_data in windows:
            try:
                window_image = window_data['image']
                bbox = window_data['bbox']
               
                # Skip if window is too small
                if window_image.shape[0] < 32 or window_image.shape[1] < 32:
                    continue
               
                # Extract features from window
                window_features = self.image_processor.extract_features(window_image)
               
                if window_features is None:
                    continue
               
                # Compute similarity with query
                similarity = self.feature_matcher.compute_similarity(
                    query_features, window_features
                )
               
                # Store detection if above threshold
                if similarity >= self.similarity_threshold:
                    detection = {
                        'x_min': bbox[0],
                        'y_min': bbox[1],
                        'x_max': bbox[2],
                        'y_max': bbox[3],
                        'searched_object_name': object_name,
                        'target_imagery_file_name': os.path.basename(satellite_image_path),
                        'similarity_score': similarity
                    }
                    detections.append(detection)
                   
            except Exception as e:
                # Continue processing other windows if one fails
                continue
       
        return detections
   
    def search_multiple_images(self, image_paths, query_features_list, object_names,
                             max_workers=4):
        """Search for objects across multiple satellite images using multithreading."""
        all_detections = []
       
        def search_single_image(args):
            image_path, query_features, object_name = args
            return self.search_in_satellite_image(image_path, query_features, object_name)
       
        # Prepare arguments for each search task
        search_tasks = []
        for image_path in image_paths:
            for query_features, object_name in zip(query_features_list, object_names):
                search_tasks.append((image_path, query_features, object_name))
       
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(search_single_image, search_tasks)
           
            for result in results:
                all_detections.extend(result)
       
        return all_detections
   
    def search_directory(self, directory_path, query_features_list, object_names):
        """Search for objects in all TIFF files in a directory."""
        # Get all TIFF files in directory
        tiff_files = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.tiff', '.tif')):
                file_path = os.path.join(directory_path, filename)
                tiff_files.append(file_path)
       
        if not tiff_files:
            print(f"No TIFF files found in {directory_path}")
            return []
       
        print(f"Found {len(tiff_files)} TIFF files to search")
       
        # Search all images
        all_detections = []
        for i, image_path in enumerate(tiff_files):
            print(f"\nProcessing image {i+1}/{len(tiff_files)}: {os.path.basename(image_path)}")
           
            for query_features, object_name in zip(query_features_list, object_names):
                detections = self.search_in_satellite_image(
                    image_path, query_features, object_name
                )
                all_detections.extend(detections)
       
        return all_detections
   
    def visualize_detections(self, image_path, detections, output_path=None):
        """Visualize detections on the satellite image."""
        try:
            # Load satellite image
            satellite_data = self.image_processor.preprocess_satellite_image(image_path)
            rgb_image = satellite_data['rgb'].copy()
           
            # Draw bounding boxes
            for detection in detections:
                x_min = int(detection['x_min'])
                y_min = int(detection['y_min'])
                x_max = int(detection['x_max'])
                y_max = int(detection['y_max'])
                score = detection['similarity_score']
               
                # Draw rectangle
                cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
               
                # Add text label
                label = f"{detection['searched_object_name']}: {score:.3f}"
                cv2.putText(rgb_image, label, (x_min, y_min - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
           
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                print(f"Visualization saved to {output_path}")
           
            return rgb_image
           
        except Exception as e:
            print(f"Error visualizing detections: {str(e)}")
            return None
   
    def evaluate_detections(self, detections, ground_truth=None):
        """Evaluate detection performance."""
        evaluation = {
            'total_detections': len(detections),
            'avg_similarity_score': 0.0,
            'score_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.6 - 0.8
                'low': 0   # < 0.6
            }
        }
       
        if detections:
            scores = [d['similarity_score'] for d in detections]
            evaluation['avg_similarity_score'] = np.mean(scores)
           
            for score in scores:
                if score > 0.8:
                    evaluation['score_distribution']['high'] += 1
                elif score > 0.6:
                    evaluation['score_distribution']['medium'] += 1
                else:
                    evaluation['score_distribution']['low'] += 1
       
        return evaluation
   
    def optimize_search_parameters(self, sample_image_path, query_features,
                                 window_sizes=[64, 128, 256],
                                 similarity_thresholds=[0.5, 0.6, 0.7, 0.8]):
        """Optimize search parameters for better performance."""
        best_params = None
        best_score = 0
       
        print("Optimizing search parameters...")
       
        for window_size in window_sizes:
            for threshold in similarity_thresholds:
                # Temporarily change parameters
                old_window_size = self.window_size
                old_threshold = self.similarity_threshold
               
                self.window_size = window_size
                self.similarity_threshold = threshold
               
                # Run search
                detections = self.search_in_satellite_image(
                    sample_image_path, query_features, "test_object"
                )
               
                # Evaluate (simple metric: number of detections with high confidence)
                high_conf_detections = len([d for d in detections if d['similarity_score'] > 0.8])
                score = high_conf_detections
               
                print(f"Window: {window_size}, Threshold: {threshold}, "
                      f"High-conf detections: {high_conf_detections}")
               
                if score > best_score:
                    best_score = score
                    best_params = {
                        'window_size': window_size,
                        'similarity_threshold': threshold
                    }
               
                # Restore old parameters
                self.window_size = old_window_size
                self.similarity_threshold = old_threshold
       
        if best_params:
            print(f"Best parameters: {best_params}")
            self.window_size = best_params['window_size']
            self.similarity_threshold = best_params['similarity_threshold']
       
        return best_params
import os
import cv2
import numpy as np
import rasterio
from PIL import Image
import tempfile
import json

def validate_tiff_file(file_path):
    """Validate if a TIFF file is a valid multispectral satellite image."""
    try:
        with rasterio.open(file_path) as src:
            # Check if file has at least 3 bands
            if src.count < 3:
                return False
           
            # Check if dimensions are reasonable
            if src.width < 100 or src.height < 100:
                return False
           
            # Check if data type is supported
            if src.dtypes[0] not in ['uint8', 'uint16', 'float32', 'float64']:
                return False
           
            return True
           
    except Exception as e:
        print(f"Error validating TIFF file {file_path}: {str(e)}")
        return False

def save_results(detections, output_file_path):
    """Save detection results to space-delimited text file."""
    try:
        with open(output_file_path, 'w') as f:
            for detection in detections:
                # Format: x_min y_min x_max y_max searched_object_name target_imagery_file_name similarity_score
                line = f"{detection['x_min']} {detection['y_min']} {detection['x_max']} {detection['y_max']} " \
                       f"{detection['searched_object_name']} {detection['target_imagery_file_name']} " \
                       f"{detection['similarity_score']:.6f}\n"
                f.write(line)
       
        print(f"Results saved to {output_file_path}")
        return True
       
    except Exception as e:
        print(f"Error saving results to {output_file_path}: {str(e)}")
        return False

def load_results(file_path):
    """Load detection results from space-delimited text file."""
    detections = []
   
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    detection = {
                        'x_min': int(parts[0]),
                        'y_min': int(parts[1]),
                        'x_max': int(parts[2]),
                        'y_max': int(parts[3]),
                        'searched_object_name': parts[4],
                        'target_imagery_file_name': parts[5],
                        'similarity_score': float(parts[6]) if parts[6] != '-1' else -1.0
                    }
                    detections.append(detection)
       
        return detections
       
    except Exception as e:
        print(f"Error loading results from {file_path}: {str(e)}")
        return []

def load_sample_images(directory_path):
    """Load sample images from a directory."""
    sample_images = []
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
   
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
           
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                try:
                    # Load image
                    if filename.lower().endswith(('.tiff', '.tif')):
                        # Handle TIFF files
                        with rasterio.open(file_path) as src:
                            if src.count >= 3:
                                # Read RGB bands
                                image = np.dstack([
                                    src.read(3),  # Red
                                    src.read(2),  # Green
                                    src.read(1)   # Blue
                                ])
                                # Normalize
                                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                            else:
                                continue
                    else:
                        # Handle regular image formats
                        image = Image.open(file_path)
                        image = np.array(image)
                       
                        # Convert to RGB if needed
                        if len(image.shape) == 3 and image.shape[2] == 4:
                            image = image[:, :, :3]  # Remove alpha channel
                        elif len(image.shape) == 2:
                            image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB
                   
                    sample_images.append({
                        'name': filename,
                        'path': file_path,
                        'image': image
                    })
                   
                except Exception as e:
                    print(f"Error loading image {filename}: {str(e)}")
                    continue
       
        return sample_images
       
    except Exception as e:
        print(f"Error loading sample images from {directory_path}: {str(e)}")
        return []

def create_image_thumbnail(image, max_size=200):
    """Create a thumbnail of an image for display."""
    try:
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape
       
        # Calculate scaling factor
        scale = min(max_size / width, max_size / height)
       
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
           
            # Resize image
            if len(image.shape) == 3:
                thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
           
            return thumbnail
        else:
            return image.copy()
           
    except Exception as e:
        print(f"Error creating thumbnail: {str(e)}")
        return image

def calculate_image_statistics(image):
    """Calculate basic statistics for an image."""
    try:
        stats = {}
       
        if len(image.shape) == 3:
            # Color image
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                channel_data = image[:, :, i].flatten()
                stats[channel] = {
                    'mean': np.mean(channel_data),
                    'std': np.std(channel_data),
                    'min': np.min(channel_data),
                    'max': np.max(channel_data)
                }
        else:
            # Grayscale image
            flat_data = image.flatten()
            stats['Grayscale'] = {
                'mean': np.mean(flat_data),
                'std': np.std(flat_data),
                'min': np.min(flat_data),
                'max': np.max(flat_data)
            }
       
        return stats
       
    except Exception as e:
        print(f"Error calculating image statistics: {str(e)}")
        return {}

def convert_coordinates(bbox, source_transform=None, target_transform=None):
    """Convert bounding box coordinates between different coordinate systems."""
    try:
        if source_transform is None or target_transform is None:
            return bbox
       
        # This is a placeholder for coordinate transformation
        # In a real implementation, you would use rasterio's transform functions
        # to convert between pixel coordinates and geographic coordinates
       
        return bbox
       
    except Exception as e:
        print(f"Error converting coordinates: {str(e)}")
        return bbox

def merge_overlapping_detections(detections, overlap_threshold=0.5):
    """Merge overlapping detections using Non-Maximum Suppression."""
    if not detections:
        return []
   
    try:
        # Convert to format suitable for NMS
        boxes = []
        scores = []
       
        for detection in detections:
            boxes.append([
                detection['x_min'],
                detection['y_min'],
                detection['x_max'],
                detection['y_max']
            ])
            scores.append(detection['similarity_score'])
       
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
       
        # Apply OpenCV's NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.1,
            nms_threshold=overlap_threshold
        )
       
        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
           
    except Exception as e:
        print(f"Error merging overlapping detections: {str(e)}")
        return detections

def generate_report(detections, output_path):
    """Generate a comprehensive report of search results."""
    try:
        report = {
            'summary': {
                'total_detections': len(detections),
                'unique_images': len(set(d['target_imagery_file_name'] for d in detections)),
                'unique_objects': len(set(d['searched_object_name'] for d in detections)),
                'avg_similarity_score': np.mean([d['similarity_score'] for d in detections]) if detections else 0
            },
            'statistics': {},
            'detections_by_image': {},
            'detections_by_object': {}
        }
       
        if detections:
            # Statistics by similarity score ranges
            scores = [d['similarity_score'] for d in detections]
            report['statistics']['score_distribution'] = {
                'high_confidence': len([s for s in scores if s > 0.8]),
                'medium_confidence': len([s for s in scores if 0.6 <= s <= 0.8]),
                'low_confidence': len([s for s in scores if s < 0.6])
            }
           
            # Group by image
            for detection in detections:
                img_name = detection['target_imagery_file_name']
                if img_name not in report['detections_by_image']:
                    report['detections_by_image'][img_name] = []
                report['detections_by_image'][img_name].append(detection)
           
            # Group by object type
            for detection in detections:
                obj_name = detection['searched_object_name']
                if obj_name not in report['detections_by_object']:
                    report['detections_by_object'][obj_name] = []
                report['detections_by_object'][obj_name].append(detection)
       
        # Save report as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
       
        print(f"Report saved to {output_path}")
        return report
       
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None

def validate_directory_structure(base_path):
    """Validate that required directories exist and create them if needed."""
    try:
        required_dirs = ['input', 'output', 'temp']
       
        for dir_name in required_dirs:
            dir_path = os.path.join(base_path, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
       
        return True
       
    except Exception as e:
        print(f"Error validating directory structure: {str(e)}")
        return False

def cleanup_temp_files(temp_dir):
    """Clean up temporary files."""
    try:
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        return True
       
    except Exception as e:
        print(f"Error cleaning up temp files: {str(e)}")
        return False

   
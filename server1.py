from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import re
import logging
from datetime import datetime

class FoodDetectionApp:
    def __init__(self, model_weights="yolo11n.pt", upload_folder="uploads", excel_file="Anuvaad_INDB_2024.11.xlsx"):
        # Initialize Flask app
        self.app = Flask(__name__, static_folder='YOLO_images', static_url_path='/YOLO_images')
        
        # Set up upload folder and allowed extensions
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        self.app.config["UPLOAD_FOLDER"] = self.upload_folder

        self.allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
        
        # Load YOLO model
        try:
            self.model = YOLO(model_weights)
            logging.info(f"YOLO model loaded successfully from {model_weights}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            raise e

        # Load food data from Excel
        self.food_data = self.load_food_data(excel_file)

        # Route Definitions
        self.app.add_url_rule('/', 'home', self.home, methods=['GET'])
        self.app.add_url_rule('/upload', 'upload_image', self.upload_image, methods=['POST'])
        self.app.add_url_rule('/detect/<filename>', 'detect_food', self.detect_food, methods=['GET'])
        self.app.add_url_rule('/uploads/<filename>', 'get_uploaded_image', self.get_uploaded_image, methods=['GET'])

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def load_food_data(self, file_path):
        """Load food data from Excel."""
        try:
            food_data = pd.read_excel(file_path, sheet_name='Sheet1')
            if 'food_name' in food_data.columns:
                food_data['food_name'] = food_data['food_name'].astype(str).str.lower()
                food_data['normalized_food_name'] = food_data['food_name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            else:
                raise KeyError("Column 'food_name' not found in Excel file.")
            logging.info(f"Food data loaded successfully from {file_path}")
            return food_data
        except FileNotFoundError:
            logging.error(f"Excel file {file_path} not found.")
            return pd.DataFrame()
        except KeyError as e:
            logging.error(f"Data error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Failed to load Excel file: {str(e)}")
            return pd.DataFrame()

    def allowed_file(self, filename):
        """Check if the file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def normalize_text(self, text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

    def get_food_details(self, food_name):
        """Fetch food details by matching the detected object's name in the food_name column."""
        if self.food_data.empty:
            logging.error("Food dataset is empty.")
            return None

        if 'food_name' not in self.food_data.columns:
            logging.error("Column 'food_name' not found in the dataset.")
            return None

        normalized_food_name = self.normalize_text(food_name)
        match = self.food_data[self.food_data['normalized_food_name'] == normalized_food_name]

        if match.empty:
            match = self.food_data[self.food_data['normalized_food_name'].str.contains(normalized_food_name, na=False)]

        if not match.empty:
            logging.info(f"Found Match for '{food_name}': {match.iloc[0]['food_name']}")
            return match.iloc[0].to_dict()
        else:
            logging.warning(f"No match found for '{food_name}' in dataset.")
        return None

    def home(self):
        return jsonify({'message': 'Welcome to the Flask YOLOv11 Food Detection API.'})

    def upload_image(self):
        if 'image' not in request.files:
            logging.error("No image file provided.")
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '' or not self.allowed_file(image_file.filename):
            logging.error("Invalid file type provided.")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Use timestamp to create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(image_file.filename)}"
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

        try:
            # Save the uploaded image with the timestamped filename
            image_file.save(filepath)
            logging.info(f"Image saved: {filepath}")
            
            # Call the /detect endpoint after the image is saved
            return self.detect_food(filename)  # Call the detection function with the saved image filename

        except Exception as e:
            logging.error(f"Error saving the file: {str(e)}")
            return jsonify({'error': f'File save failed: {str(e)}'}), 500

    def detect_food(self, filename):
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            logging.error(f"Image not found: {filepath}")
            return jsonify({'error': 'Image not found'}), 404

        try:
            image = Image.open(filepath)  # Open the image using PIL
        except UnidentifiedImageError:
            logging.error(f"Image could not be identified: {filepath}")
            return jsonify({'error': 'Invalid image format'}), 400
        except Exception as e:
            logging.error(f"Error opening image: {str(e)}")
            return jsonify({'error': f'Error opening image: {str(e)}'}), 500
        
        try:
            results = self.model(image)  # Run the YOLO detection on the image
        except Exception as e:
            logging.error(f"Error processing the image with YOLO: {str(e)}")
            return jsonify({'error': f'Failed to process image with YOLO: {str(e)}'}), 500

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id].strip()
                logging.info(f"Detected Class: {class_name}")
                food_details = self.get_food_details(class_name)

                detection_info = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                    "food_details": food_details if food_details else "Not found in database"
                }
                detections.append(detection_info)

        return jsonify({
            'message': 'Object detection completed',
            'detections': detections
        }), 200

    def get_uploaded_image(self, filename):
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(filepath):
            logging.error(f"Requested file not found: {filepath}")
            return jsonify({'error': f'File {filename} not found'}), 404

        try:
            return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)
        except Exception as e:
            logging.error(f"Error sending file: {str(e)}")
            return jsonify({'error': f'Failed to retrieve file: {str(e)}'}), 500

    def run(self):
        self.app.run(host='0.0.0.0', port=5001, debug=True)


if __name__ == '__main__':
    app_instance = FoodDetectionApp()
    app_instance.run()

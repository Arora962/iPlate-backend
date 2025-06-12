from flask import Flask, request, jsonify
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import re
import logging
from datetime import datetime
from collections import defaultdict
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid
import json
import torch

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

load_dotenv()

# Initialize Firebase Admin
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
firebase_admin.initialize_app(cred)

class FoodDetectionApp:
    def __init__(self, model_weights="yolo11n.pt", upload_folder="uploads", supabase_url="your-supabase-url", supabase_key="your-supabase-key"):
        self.app = Flask(__name__)
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        self.app.config["UPLOAD_FOLDER"] = self.upload_folder
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        self.supabase: Client = create_client(supabase_url, supabase_key)

        try:
            from ultralytics.nn.tasks import DetectionModel

            with torch.serialization.safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel}):
                self.model = YOLO(model_weights)

            logging.info(f"YOLO model loaded from {model_weights}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            raise e


        self.food_data = self.load_food_data()
        logging.basicConfig(level=logging.INFO)

        # Routes
        self.app.add_url_rule('/', 'home', self.home, methods=['GET'])
        self.app.add_url_rule('/meals', 'create_meal', self.create_meal, methods=['POST'])
        self.app.add_url_rule('/meals/<meal_id>/items', 'add_meal_item', self.add_meal_item, methods=['POST'])
        self.app.add_url_rule('/meals/<meal_id>/items', 'get_meal_items', self.get_meal_items, methods=['GET'])
        self.app.add_url_rule('/upload', 'upload', self.upload, methods=['POST'])
        self.app.add_url_rule('/meals/previous', 'get_previous_meal', self.get_previous_meal, methods=['GET'])
        self.app.add_url_rule('/meals/grouped', 'get_grouped_meals', self.get_grouped_meals, methods=['GET'])

    def verify_firebase_token(self, id_token):
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            return decoded
        except Exception as e:
            logging.warning(f"Firebase token verification failed: {e}")
            return None

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def normalize_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

    def load_food_data(self):
        try:
            res = self.supabase.table("food_data").select("*").execute()
            df = pd.DataFrame(res.data)
            df['food_name'] = df['food_name'].astype(str).str.lower()
            df['normalized_food_name'] = df['food_name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
            return df
        except Exception as e:
            logging.error(f"Error loading food_data: {e}")
            return pd.DataFrame()

    def home(self):
        return jsonify({"message": "Welcome to the Flask YOLOv11 Food Detection API."})

    def create_meal(self):
        data = request.get_json()
        user_id = data.get("user_id")
        meal_type = data.get("meal_type")
        date = data.get("date", datetime.utcnow().date().isoformat())

        res = self.supabase.table("meals").insert({
            "user_id": user_id,
            "meal_type": meal_type,
            "date": date
        }).execute()

        return jsonify({"message": "Meal created", "meal_id": res.data[0]["id"]}), 201

    def add_meal_item(self, meal_id):
        data = request.get_json()
        food_code = data.get("food_code")
        quantity_grams = data.get("quantity_grams")

        food = self.supabase.table("food_data").select("*").eq("food_code", food_code).execute().data
        if not food:
            return jsonify({"error": "Food not found"}), 404

        food = food[0]
        factor = quantity_grams / 100.0

        item = {
            "meal_id": meal_id,
            "food_code": food_code,
            "quantity_grams": quantity_grams,
            "energy": food["energy_kj"] * factor,
            "calories": food["energy_kcal"] * factor,
            "protein": food["protein_g"] * factor,
            "carbs": food["carb_g"] * factor,
            "fat": food["fat_g"] * factor,
            "fiber": food["fibre_g"] * factor
        }

        res = self.supabase.table("meal_items").insert(item).execute()
        return jsonify({"message": "Meal item added", "item": res.data[0]}), 201

    def get_meal_items(self, meal_id):
        items = self.supabase.table("meal_items").select("*").eq("meal_id", meal_id).execute().data
        if not items:
            return jsonify({"message": "No items found for this meal", "items": [], "summary": {}})

        summary = {
            "energy": sum(i.get("energy", 0) for i in items),
            "calories": sum(i.get("calories", 0) for i in items),
            "protein": sum(i.get("protein", 0) for i in items),
            "carbs": sum(i.get("carbs", 0) for i in items),
            "fat": sum(i.get("fat", 0) for i in items),
            "fiber": sum(i.get("fiber", 0) for i in items)
        }

        return jsonify({"items": items, "summary": summary})

    def upload(self):
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        image_file = request.files['image']
        if image_file.filename == '' or not self.allowed_file(image_file.filename):
            return jsonify({"error": "Invalid image type"}), 400

        weight_input = request.form.get("weights")
        if not weight_input:
            return jsonify({"error": "Missing weight input"}), 400

        try:
            weight_values = [float(w.strip()) for w in weight_input.split(",")]
        except ValueError:
            return jsonify({"error": "Invalid weight format."}), 400

        # Unified Firebase Auth check
        firebase_uid = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            id_token = auth_header.split(" ")[1]
            try:
                decoded = self.verify_firebase_token(id_token)
                firebase_uid = decoded["uid"]
            except Exception as e:
                logging.warning(f"ID token verification failed: {e}")
                return jsonify({"error": "Invalid Firebase token"}), 401
            
        # Development-mode fallback using X-Firebase-UID
        elif os.getenv("FLASK_ENV") == "development":
            firebase_uid = request.headers.get("X-Firebase-UID")
            if not firebase_uid:
                return jsonify({"error": "Missing Firebase UID (X-Firebase-UID)"}), 401
        else:
            return jsonify({"error": "Authorization required"}), 401

        email = decoded.get("email") if auth_header else None

        # Get or create user in Supabase
        user_res = self.supabase.table("users").select("id").eq("firebase_uid", firebase_uid).execute()
        if not user_res.data:
            insert_res = self.supabase.table("users").insert({
                "firebase_uid": firebase_uid,
                "email": email
            }).execute()
            user_id = insert_res.data[0]["id"]
        else:
            user_id = user_res.data[0]["id"]

        def infer_meal_type():
            hour = datetime.utcnow().hour
            if hour < 11:
                return "breakfast"
            elif hour < 16:
                return "lunch"
            else:
                return "dinner"

        meal_type = infer_meal_type()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(image_file.filename)}"
        filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)

        # Create meal
        meal_res = self.supabase.table("meals").insert({
            "user_id": user_id,
            "meal_type": meal_type,
            "date": datetime.utcnow().date().isoformat()
        }).execute()
        meal_id = meal_res.data[0]['id']

        # Run YOLO detection
        try:
            image = Image.open(filepath)
            results = self.model(image)
        except Exception as e:
            return jsonify({"error": f"Detection failed: {str(e)}"}), 500

        all_detections = []

        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])].strip()
                normalized_name = self.normalize_text(class_name)
                match = self.food_data[self.food_data['normalized_food_name'].str.contains(normalized_name, na=False)]

                if not match.empty:
                    food = match.iloc[0]
                    all_detections.append((food, class_name, normalized_name))

        if not all_detections:
            return jsonify({"message": "No food items matched", "items": [], "summary": {}})

        if len(weight_values) == 1:
            if len(all_detections) != 1:
                return jsonify({
                    "error": "Single weight value provided but multiple food items detected. Provide one weight per item."
                }), 400
            weight_list = weight_values
        elif len(weight_values) == len(all_detections):
            weight_list = weight_values
        else:
            return jsonify({
                "error": f"Mismatch between weights ({len(weight_values)}) and detected items ({len(all_detections)})."
            }), 400

        # Insert meal items
        detections = []
        for (food, class_name, _), grams in zip(all_detections, weight_list):
            factor = grams / 100.0
            item = {
                "meal_id": meal_id,
                "food_code": food['food_code'],
                "quantity_grams": grams,
                "energy": food["energy_kj"] * factor,
                "calories": food["energy_kcal"] * factor,
                "protein": food["protein_g"] * factor,
                "carbs": food["carb_g"] * factor,
                "fat": food["fat_g"] * factor,
                "fiber": food["fibre_g"] * factor,
                "class_name": class_name
            }

            self.supabase.table("meal_items").insert({k: v for k, v in item.items() if k != "class_name"}).execute()
            detections.append(item)

        summary = {
            "energy": sum(i["energy"] for i in detections),
            "calories": sum(i["calories"] for i in detections),
            "protein": sum(i["protein"] for i in detections),
            "carbs": sum(i["carbs"] for i in detections),
            "fat": sum(i["fat"] for i in detections),
            "fiber": sum(i["fiber"] for i in detections)
        }

        summary_items = [
            {
                "food": item["class_name"],
                "quantity_grams": item["quantity_grams"]
            }
            for item in detections
        ]

        return jsonify({
            "message": "Meal summary created",
            "meal_id": meal_id,
            "foods": summary_items,
            "summary": summary
        })
    
    def get_previous_meal(self):
        firebase_uid = request.headers.get("X-Firebase-UID") if os.getenv("FLASK_ENV") == "development" else None
        if not firebase_uid:
            return jsonify({"error": "Missing Firebase UID"}), 401

        user_res = self.supabase.table("users").select("id").eq("firebase_uid", firebase_uid).execute()
        if not user_res.data:
            return jsonify({"error": "User not found"}), 404

        user_id = user_res.data[0]["id"]

        # Get most recent meal
        meal_res = self.supabase.table("meals") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if not meal_res.data:
            return jsonify({"message": "No meals found"}), 404

        meal = meal_res.data[0]
        meal_id = meal["id"]

        items = self.supabase.table("meal_items").select("*").eq("meal_id", meal_id).execute().data

        summary = {
            "energy": sum(i.get("energy", 0) for i in items),
            "calories": sum(i.get("calories", 0) for i in items),
            "protein": sum(i.get("protein", 0) for i in items),
            "carbs": sum(i.get("carbs", 0) for i in items),
            "fat": sum(i.get("fat", 0) for i in items),
            "fiber": sum(i.get("fiber", 0) for i in items)
        }

        weights = [
            {
                "food_code": i.get("food_code"),
                "quantity_grams": i.get("quantity_grams")
            }
            for i in items
        ]

        return jsonify({
            "summary": summary,
            "weights": weights
        })


    def run(self):
        self.app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == "__main__":
    app_instance = FoodDetectionApp(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    app_instance.run()

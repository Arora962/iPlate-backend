# Server with Postgres DB (Dataset - Anuvaad)

# server.py

from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import re
import logging
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import bcrypt
import uuid
import json

load_dotenv()

class FoodDetectionApp:
    def __init__(self, model_weights="yolo11n.pt", upload_folder="uploads", supabase_url="your-supabase-url", supabase_key="your-supabase-key"):
        self.app = Flask(__name__)
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        self.app.config["UPLOAD_FOLDER"] = self.upload_folder
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        self.supabase: Client = create_client(supabase_url, supabase_key)

        try:
            self.model = YOLO(model_weights)
            logging.info(f"YOLO model loaded from {model_weights}")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {str(e)}")
            raise e

        self.food_data = self.load_food_data()
        logging.basicConfig(level=logging.INFO)

        # Routes
        self.app.add_url_rule('/', 'home', self.home, methods=['GET'])
        self.app.add_url_rule('/signup', 'signup', self.signup, methods=['POST'])
        self.app.add_url_rule('/login', 'login', self.login, methods=['POST'])
        self.app.add_url_rule('/meals', 'create_meal', self.create_meal, methods=['POST'])
        self.app.add_url_rule('/meals/<meal_id>/items', 'add_meal_item', self.add_meal_item, methods=['POST'])
        self.app.add_url_rule('/meals/<meal_id>/items', 'get_meal_items', self.get_meal_items, methods=['GET'])
        self.app.add_url_rule('/upload_and_add_items', 'upload_and_add_items', self.upload_and_add_items, methods=['POST'])

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password, hashed):
        return bcrypt.checkpw(password.encode(), hashed.encode())

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

    def signup(self):
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Missing email or password'}), 400

        existing = self.supabase.table("users").select("*").eq("email", email).execute()
        if existing.data:
            return jsonify({'error': 'User already exists'}), 409

        hashed = self.hash_password(password)
        token = str(uuid.uuid4())

        self.supabase.table("users").insert({
            "email": email,
            "password_hash": hashed,
            "session_token": token
        }).execute()

        return jsonify({"message": "User created", "session_token": token}), 201

    def login(self):
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        user = self.supabase.table("users").select("*").eq("email", email).execute().data
        if not user:
            return jsonify({'error': 'User not found'}), 404

        user = user[0]
        if not self.verify_password(password, user['password_hash']):
            return jsonify({'error': 'Invalid credentials'}), 401

        token = str(uuid.uuid4())
        self.supabase.table("users").update({"session_token": token}).eq("id", user["id"]).execute()
        return jsonify({"message": "Login successful", "user_id": user["id"], "session_token": token})

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
            "calories": sum(i.get("calories", 0) for i in items),
            "protein": sum(i.get("protein", 0) for i in items),
            "carbs": sum(i.get("carbs", 0) for i in items),
            "fat": sum(i.get("fat", 0) for i in items),
            "fiber": sum(i.get("fiber", 0) for i in items)
        }

        return jsonify({"items": items, "summary": summary})

    def upload_and_add_items(self):
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        image_file = request.files['image']
        if image_file.filename == '' or not self.allowed_file(image_file.filename):
            return jsonify({"error": "Invalid image type"}), 400

        user_id = request.form.get('user_id')
        meal_type = request.form.get('meal_type')
        weights_str = request.form.get('weights')  # JSON string

        if not user_id or not meal_type or not weights_str:
            return jsonify({"error": "Missing user_id, meal_type or weights"}), 400

        try:
            weights = json.loads(weights_str)
        except Exception:
            return jsonify({"error": "Invalid weights JSON format"}), 400

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

        # Run detection
        try:
            image = Image.open(filepath)
            results = self.model(image)
        except Exception as e:
            return jsonify({"error": f"Detection failed: {str(e)}"}), 500

        detections = []
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls[0])].strip()
                normalized_name = self.normalize_text(class_name)
                match = self.food_data[self.food_data['normalized_food_name'].str.contains(normalized_name, na=False)]

                if not match.empty:
                    food = match.iloc[0]
                    grams = weights.get(normalized_name)
                    if not grams:
                        continue

                    factor = grams / 100.0
                    item = {
                        "meal_id": meal_id,
                        "food_code": food['food_code'],
                        "quantity_grams": grams,
                        "calories": food["energy_kcal"] * factor,
                        "protein": food["protein_g"] * factor,
                        "carbs": food["carb_g"] * factor,
                        "fat": food["fat_g"] * factor,
                        "fiber": food["fibre_g"] * factor,
                        "class_name": class_name
                    }

                    # Insert only fields that exist in the Supabase meal_items table
                    self.supabase.table("meal_items").insert({k: v for k, v in item.items() if k != "class_name"}).execute()

                    # Add class_name only to the returned data (not DB)
                    item["class_name"] = class_name
                    detections.append(item)


        if not detections:
            return jsonify({"message": "No food items matched with given weights", "items": [], "summary": {}})

        summary = {
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

    def run(self):
        self.app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == "__main__":
    app_instance = FoodDetectionApp(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    app_instance.run()

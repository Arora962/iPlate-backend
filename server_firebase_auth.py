from flask import Flask, request, jsonify
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import re
import logging
from collections import defaultdict
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
import uuid
import json
import torch
import requests
import pytz
import random
from datetime import datetime, time
from io import BytesIO
from ultralytics.nn.tasks import DetectionModel

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

load_dotenv()

# Initialize Firebase Admin
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
firebase_admin.initialize_app(cred)

from shapely.geometry import Point, Polygon

def sort_detections_by_regions(detections, image_size):
    """
    Sorts the detections into regions based on a predefined layout.
    Regions:
        1 - bottom half
        2 - top-left
        3 - top-middle
        4 - top-right
    """
    width, height = image_size
    regions = {
        1: Polygon([(0, height * 0.5), (width, height * 0.5), (width, height), (0, height)]),
        2: Polygon([(0, 0), (width / 3, 0), (width / 3, height * 0.5), (0, height * 0.5)]),
        3: Polygon([(width / 3, 0), (2 * width / 3, 0), (2 * width / 3, height * 0.5), (width / 3, height * 0.5)]),
        4: Polygon([(2 * width / 3, 0), (width, 0), (width, height * 0.5), (2 * width / 3, height * 0.5)])
    }

    region_map = {1: [], 2: [], 3: [], 4: []}

    for detection in detections:
        box = detection["box"]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        center = Point(cx, cy)

        for region_id, polygon in regions.items():
            if polygon.contains(center):
                region_map[region_id].append((cx, detection))
                break

    sorted_detections = []
    if region_map[1]:
        sorted_detections.extend([det[1] for det in region_map[1]])  # First detection from region 1
    for region_id in [2, 3, 4]:
        sorted_items = sorted(region_map[region_id], key=lambda x: x[0])
        sorted_detections.extend([det[1] for det in sorted_items])

    return sorted_detections


class FoodDetectionApp:
    def __init__(self, model_weights="yolo11n.pt", upload_folder="uploads", supabase_url="your-supabase-url", supabase_key="your-supabase-key"):
        self.app = Flask(__name__)
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)
        self.app.config["UPLOAD_FOLDER"] = self.upload_folder
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        self.supabase: Client = create_client(supabase_url, supabase_key)

        try:
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
        self.app.add_url_rule('/auth/forgot-password', 'send_password_reset', self.send_password_reset, methods=['POST'])
        self.app.add_url_rule('/mood/suggestions', 'get_mood_suggestions_with_input', self.get_mood_suggestions_with_input, methods=['POST'])

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
        image_files = request.files.getlist("image")
        if not image_files:
            return jsonify({"error": "At least one image file is required"}), 400

        for file in image_files:
            if file.filename == '' or not self.allowed_file(file.filename):
                return jsonify({"error": f"Invalid file type: {file.filename}"}), 400

        weight_input = request.form.get("weights")
        if not weight_input:
            return jsonify({"error": "Missing weight input"}), 400

        try:
            weight_values = [float(w.strip()) for w in weight_input.split(",")]
        except ValueError:
            return jsonify({"error": "Invalid weight format."}), 400

        # Firebase Authentication
        auth_header = request.headers.get("Authorization")
        firebase_uid = None
        email = None
        decoded = None

        if auth_header and auth_header.startswith("Bearer "):
            id_token = auth_header.split(" ")[1]
            decoded = self.verify_firebase_token(id_token)
            if not decoded:
                return jsonify({"error": "Invalid Firebase token"}), 401
            firebase_uid = decoded.get("uid")
            email = decoded.get("email")

        elif os.getenv("FLASK_ENV") == "development":
            firebase_uid = request.headers.get("X-Firebase-UID")
            email = request.headers.get("X-Firebase-Email")
        else:
            return jsonify({"error": "Authorization required"}), 401

        # Validate Firebase UID and Email
        if not firebase_uid or not email:
            return jsonify({"error": "Missing Firebase UID or email"}), 400

        logging.info(f"Firebase UID: {firebase_uid}, Email: {email}")

        # Check if the user already exists in Supabase
        try:
            user_res = self.supabase.table("users").select("id").eq("firebase_uid", firebase_uid).execute()
            if user_res.data:
                # User exists, get the user_id
                user_id = user_res.data[0]["id"]
            else:
                # User doesn't exist, insert the new user and get the user_id
                insert_res = self.supabase.table("users").insert({
                    "firebase_uid": firebase_uid,
                    "email": email
                }).execute()
                user_id = insert_res.data[0]["id"]
        except Exception as e:
            logging.error(f"User lookup or insert failed: {e}")
            return jsonify({"error": "Failed to fetch or create user", "details": str(e)}), 500

        def infer_meal_type():
            # Use local timezone (IST)
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist).time()

            def in_range(start_hour, start_minute, end_hour, end_minute):
                return time(start_hour, start_minute) <= now <= time(end_hour, end_minute)

            if in_range(7, 0, 9, 30):
                return "breakfast"
            elif in_range(10, 0, 11, 0):
                return "morning snack"
            elif in_range(12, 0, 14, 0):
                return "lunch"
            elif in_range(16, 0, 17, 30):
                return "evening snack"
            elif in_range(19, 0, 22, 0):
                return "dinner"
            else:
                return "other"

        meal_type = infer_meal_type()
        all_detections = []
        image_metadata = [] # Stores image, stream, filename

        # Read all images into memory
        for image_file in image_files:
            image_stream = BytesIO(image_file.read())
            image = Image.open(image_stream)
            filename = secure_filename(image_file.filename)
            image_metadata.append({
                "image": image,
                "stream": image_stream,
                "filename": filename
            })

        # Detect food in each image
        for img_data in image_metadata:
            image = img_data["image"]
            filename = img_data["filename"]
            try:
                results = self.model(image)
            except Exception as e:
                return jsonify({"error": f"Detection failed on {filename}: {str(e)}"}), 500

            image_width, image_height = image.size
            raw_detections = []

            for result in results:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls[0])].strip()
                    normalized_name = self.normalize_text(class_name)
                    xyxy = box.xyxy[0].tolist()

                    match = self.food_data[self.food_data['normalized_food_name'] == normalized_name]
                    if match.empty:
                        match = self.food_data[self.food_data['normalized_food_name'].str.contains(normalized_name, na=False)]

                    if not match.empty:
                        food = match.iloc[0]
                        raw_detections.append({
                            "food": food,
                            "food_name": class_name,
                            "normalized_name": normalized_name,
                            "filename": filename,
                            "stream": img_data["stream"],
                            "box": xyxy
                        })

            # Sort detections based on plate layout
            sorted_detections = sort_detections_by_regions(raw_detections, (image_width, image_height))
            all_detections.extend(sorted_detections)
        
        # Weight assignment validation
        if len(weight_values) == 1 and len(all_detections) > 1:
            weight_list = [weight_values[0]] * len(all_detections)
        elif len(weight_values) == len(all_detections):
            weight_list = weight_values
        else:
            return jsonify({
                "error": f"Mismatch between weights ({len(weight_values)}) and detected items ({len(all_detections)})."
            }), 400

        # Creation of meal record
        try:
            meal_res = self.supabase.table("meals").insert({
                "user_id": user_id,
                "meal_type": meal_type,
                "date": datetime.utcnow().date().isoformat()
            }).execute()
            meal_id = meal_res.data[0]['id']
        except Exception as e:
            return jsonify({"error": f"Failed to create meal: {str(e)}"}), 500

        # Insertion of meal items
        detections = []
        try:
            for detection, grams in zip(all_detections, weight_list):
                food = detection["food"]
                class_name = detection["food_name"]
                filename = detection["filename"]
                factor = grams / 100.0
                item = {
                    "meal_id": meal_id,
                    "food_code": food.get("food_code"),
                    "food_name": class_name,
                    "quantity_grams": grams,
                    "energy": (food.get("energy_kj") or 0) * factor,
                    "calories": (food.get("energy_kcal") or 0) * factor,
                    "protein": (food.get("protein_g") or 0) * factor,
                    "carbs": (food.get("carb_g") or 0) * factor,
                    "fat": (food.get("fat_g") or 0) * factor,
                    "fiber": (food.get("fibre_g") or 0) * factor
                }

                self.supabase.table("meal_items").insert(item).execute()
                detections.append(item)

            # Save each image after successful insertion
            for img_data in image_metadata:
                stream = img_data["stream"]
                stream.seek(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{img_data['filename']}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)

                with open(filepath, 'wb') as f:
                    f.write(stream.read())

        except Exception as insert_err:
            self.supabase.table("meals").delete().eq("id", meal_id).execute()
            return jsonify({
                "error": "Failed to insert meal items.",
                "details": str(insert_err)
            }), 500

        # Building summary of data for food item(s)
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
                "food": item["food_name"],
                "quantity_grams": item["quantity_grams"],
                "calories": item["calories"],
                "protein": item["protein"],
                "carbs": item["carbs"],
                "fat": item["fat"],
                "fiber": item["fiber"],
                "energy": item["energy"]
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
            "fiber": sum(i.get("fiber", 0) for i in items),
        }

        quantity_total = sum(i.get("quantity_grams", 0) for i in items)

        food_breakdown = [
            {
                "food_name": i.get("food_name"),
                "quantity_grams": i.get("quantity_grams", 0),
                "calories": i.get("calories", 0),
                "protein": i.get("protein", 0),
                "carbs": i.get("carbs", 0),
                "fat": i.get("fat", 0),
                "fiber": i.get("fiber", 0)
            }
            for i in items
        ]

        response = {
            "meal_id": meal_id,
            "quantity_total": quantity_total,
            "summary": summary
        }

        if len(items) > 1:
            response["Item List"] = food_breakdown

        return jsonify(response)

    def get_grouped_meals(self):
        firebase_uid = request.headers.get("X-Firebase-UID") if os.getenv("FLASK_ENV") == "development" else None
        if not firebase_uid:
            return jsonify({"error": "Missing Firebase UID"}), 401

        user_res = self.supabase.table("users").select("id").eq("firebase_uid", firebase_uid).execute()
        if not user_res.data:
            return jsonify({"error": "User not found"}), 404

        user_id = user_res.data[0]["id"]

        # Use created_at, not date, for real chronological grouping
        meals_res = self.supabase.table("meals") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(1000) \
            .execute()

        meals = meals_res.data

        # Group by formatted created_at date
        grouped_by_date = defaultdict(lambda: {
            "breakfast": [],
            "morning snack": [],
            "lunch": [],
            "evening snack": [],
            "dinner": []
        })

        for meal in meals:
            created_at_str = meal.get("created_at")
            if not created_at_str:
                continue # skipping invalid entry

            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            date_str = created_at.strftime("%d-%m-%Y") # dd-mm-yyyy format

            meal_type = meal.get("meal_type", "").strip().lower()
            if meal_type not in ("breakfast", "morning snack", "lunch", "evening snack", "dinner"):
                continue # skipping unknown types

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

            quantity_total = sum(i.get("quantity_grams", 0) for i in items)

            meal_data = {
                "meal_id": meal_id,
                "quantity_total": quantity_total,
                "summary": summary
            }

            if len(items) > 1:
                meal_data["Item List"] = [
                    {
                        "food_name": i.get("food_name"),
                        "quantity_grams": i.get("quantity_grams", 0),
                        "calories": i.get("calories", 0),
                        "protein": i.get("protein", 0),
                        "carbs": i.get("carbs", 0),
                        "fat": i.get("fat", 0),
                        "fiber": i.get("fiber", 0)
                    }
                    for i in items
                ]

            grouped_by_date[date_str][meal_type].append(meal_data)
        
        # Final ordered list for output (latest date first)
        sorted_dates = sorted(
            grouped_by_date.keys(),
            key=lambda d: datetime.strptime(d, "%d-%m-%Y"),
            reverse=True
        )

        ordered_output = []
        for date in sorted_dates:
            meals_by_type = grouped_by_date[date]
            ordered_output.append({
                "date": date,
                "meals": {
                    "breakfast": meals_by_type["breakfast"],
                    "morning snack": meals_by_type["morning snack"],
                    "lunch": meals_by_type["lunch"],
                    "evening snack": meals_by_type["evening snack"],
                    "dinner": meals_by_type["dinner"]
                }
            })

        return jsonify(ordered_output)

    def send_password_reset(self):
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"error": "Email is required"}), 400

        try:
            firebase_api_key = os.getenv("FIREBASE_WEB_API_KEY")
            if not firebase_api_key:
                raise ValueError("Missing FIREBASE_WEB_API_KEY in environment variables.")

            reset_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={firebase_api_key}"

            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }

            response = requests.post(reset_url, json=payload)

            if response.status_code == 200:
                return jsonify({"message": "Password reset email sent successfully."}), 200
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                return jsonify({"error": f"Failed to send reset email: {error_msg}"}), 400

        except Exception as e:
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
    def get_mood_suggestions_with_input(self):
        try:
            data = request.get_json()

            # Parse input
            diet_tags = data.get("diet_tags", "")
            allergens = data.get("allergens", "")
            current_mood = data.get("current_mood", "")

            if not current_mood:
                return jsonify({"error": "current_mood is required"}), 400

            # Normalize input
            input_diet_tags = set(tag.strip().lower() for tag in diet_tags.split(",") if tag.strip())
            
            def parse_allergen_string(allergen_str):
                result = set()
                for part in allergen_str.split(","):
                    for subpart in part.split("/"):
                        cleaned = subpart.strip().lower()
                        if cleaned:
                            result.add(cleaned)
                return result

            input_allergens = parse_allergen_string(allergens)

            input_mood = current_mood.strip().lower()

            # Determine current time (IST) → meal time
            now = datetime.now(pytz.timezone("Asia/Kolkata")).time()

            def in_range(start, end):
                return start <= now <= end

            if in_range(time(7, 0), time(9, 30)):
                meal_time = "breakfast"
            elif in_range(time(10, 0), time(11, 0)):
                meal_time = "morning snack"
            elif in_range(time(12, 0), time(15, 0)):
                meal_time = "lunch"
            elif in_range(time(16, 0), time(17, 30)):
                meal_time = "evening snack"
            elif in_range(time(19, 0), time(22, 0)):
                meal_time = "dinner"
            else:
                meal_time = None

            if not meal_time:
                return jsonify({"message": "No suitable meal time for current hour."}), 200

            # Save mood input
            insert_res = self.supabase.table("mood_inputs").insert({
                "diet_tags": diet_tags,
                "allergens": allergens,
                "current_mood": current_mood
            }).execute()

            # Fetch food entries from mood_table
            mood_data = self.supabase.table("mood_table").select("*").execute().data
            strict_matches = []
            partial_matches = []

            for food in mood_data:
                food_name = food.get("food_name", "").strip()
                description = food.get("description", "").strip()
                best_time = (food.get("best_time_to_eat") or "").strip().lower()

                food_diet_tags = set(tag.strip().lower() for tag in (food.get("diet_tags") or "").split(",") if tag.strip())
                food_mood_tags = set(tag.strip().lower() for tag in (food.get("mood_tags") or "").split(",") if tag.strip())
                food_allergens = set(tag.strip().lower() for tag in (food.get("allergens") or "").split(",") if tag.strip())

                # Filter: allergens
                if input_allergens & food_allergens:
                    continue

                # Filter: mood
                if input_mood not in food_mood_tags:
                    continue

                # Filter: time
                if best_time != "any" and meal_time not in best_time:
                    continue

                # Filter: diet tag logic
                if len(input_diet_tags) == 1:
                    single_tag = next(iter(input_diet_tags))
                    if food_diet_tags == input_diet_tags:
                        strict_matches.append({
                            "food_name": food_name,
                            "description": description
                        })
                    elif single_tag in food_diet_tags:
                        partial_matches.append({
                            "food_name": food_name,
                            "description": description
                        })
                elif len(input_diet_tags) > 1:
                    if input_diet_tags.issubset(food_diet_tags):
                        strict_matches.append({
                            "food_name": food_name,
                            "description": description
                        })
                    elif input_diet_tags & food_diet_tags:
                        partial_matches.append({
                            "food_name": food_name,
                            "description": description
                        })

            # Combine results
            final_suggestions = strict_matches + partial_matches
            random.shuffle(final_suggestions)

            if not final_suggestions:
                return jsonify({"message": "No suitable food suggestions found."}), 200

            return jsonify({
                "message": "Suggested foods based on your mood",
                "suggestions": final_suggestions[:5]
            }), 200

        except Exception as e:
            return jsonify({"error": f"Server error: {str(e)}"}), 500

    def run(self):
        self.app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == "__main__":
    app_instance = FoodDetectionApp(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    app_instance.run()

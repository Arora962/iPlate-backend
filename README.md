# Iplate
Iplate Server Code

Features
YOLO-based License Plate Detection:
Utilizes YOLOv8 models (yolo11n.pt, yolov8n.pt) for accurate and efficient license plate recognition from images.

Flexible Data Storage:
Supports storing extracted license plate data in CSV and Excel formats (Db.csv, Db.xlsx, Anuvaad_INDB_2024.11.xlsx).

Multiple Server Modes:
Includes different server scripts for varied storage backends:

server.py: General-purpose image processing and data handling

server_csv.py: Focused on CSV storage

server_db.py: Focused on Excel/database storage

Image Upload Handling:
Accepts and processes images via the uploads/ directory.

Environment Configuration:
Provides an example .env_example file for easy setup of environment variables.


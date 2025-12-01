"""
Complete Flask Backend for Crop Recognition
Run this on your server/computer
"""
#pip install flask flask-cors torch torchvision transformers pillow
#python app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from any origin (your website)

# Global variable for model
recognizer = None

class CropRecognizer:
    """Crop recognition using Hugging Face Vision Transformers"""
    
    def __init__(self, model_name="wambugu71/crop_leaf_diseases_vit"):
        logger.info(f"Loading model: {model_name}")
        
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name)
            logger.info("✅ Crop-specific model loaded!")
        except Exception as e:
            logger.error(f"Error: {e}. Using general model...")
            model_name = "google/vit-base-patch16-224"
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(model_name)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        self.model_name = model_name
    
    def predict(self, image_bytes, top_k=5):
        """Predict crop from image bytes"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            
            # Format results
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx.item()]
                predictions.append({
                    "crop_name": label,
                    "confidence": round(prob.item() * 100, 2)
                })
            
            return {
                "success": True,
                "primary_crop": predictions[0]["crop_name"],
                "confidence": predictions[0]["confidence"],
                "all_predictions": predictions,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Initialize model when app starts
@app.before_request
def load_model():
    global recognizer
    if recognizer is None:
        logger.info("🚀 Initializing Crop Recognizer...")
        recognizer = CropRecognizer()
        logger.info("✅ Ready to accept requests!")


# API Endpoints
@app.route('/')
def index():
    """Serve the frontend HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop Recognition API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #333; }
            .endpoint {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
                border-radius: 5px;
            }
            code {
                background: #e9ecef;
                padding: 2px 5px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌾 Crop Recognition API</h1>
            <p>Your API is running successfully!</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <strong>POST /recognize</strong><br>
                Upload image file for crop recognition<br>
                <code>Content-Type: multipart/form-data</code>
            </div>
            
            <div class="endpoint">
                <strong>POST /recognize-base64</strong><br>
                Send base64 encoded image<br>
                <code>Content-Type: application/json</code>
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong><br>
                Check API health status
            </div>
            
            <p style="margin-top: 30px;">
                <strong>Next Step:</strong> Use the mobile-friendly frontend HTML to interact with this API.
            </p>
        </div>
    </body>
    </html>
    """


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": recognizer.model_name if recognizer else "not loaded",
        "device": str(recognizer.device) if recognizer else "unknown"
    })


@app.route('/recognize', methods=['POST'])
def recognize_crop():
    """
    Recognize crop from uploaded image file
    Expects: multipart/form-data with 'image' field
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Send image in 'image' field."
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400
        
        # Read image
        image_bytes = file.read()
        
        # Predict
        result = recognizer.predict(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recognize-base64', methods=['POST'])
def recognize_base64():
    """
    Recognize crop from base64 encoded image
    Expects: JSON with 'image' field containing base64 string
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data. Send JSON with 'image' field."
            }), 400
        
        base64_string = data['image']
        
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        # Decode base64
        import base64
        image_bytes = base64.b64decode(base64_string)
        
        # Predict
        result = recognizer.predict(image_bytes)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    # Get port from environment or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("🌾 CROP RECOGNITION API SERVER")
    print("="*60)
    print(f"✅ Server starting on http://localhost:{port}")
    print(f"✅ Health check: http://localhost:{port}/health")
    print(f"✅ Main endpoint: http://localhost:{port}/recognize")
    print("="*60 + "\n")
    
    # Run server
    app.run(
        host='0.0.0.0',  # Accessible from any IP
        port=port,
        debug=True
    )
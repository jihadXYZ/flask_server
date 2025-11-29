# 🌾 Crop Recognition API

A powerful, AI-driven REST API for automatic crop identification using Vision Transformers. Upload an image of a crop or plant leaf, and get instant predictions with confidence scores.

**[Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Deployment](#deployment) • [Contributing](#contributing)**

---

## ✨ Features

- **🎯 Accurate Crop Recognition** - Powered by Vision Transformer (ViT) models fine-tuned for crop identification
- **⚡ Real-time Predictions** - GPU-accelerated inference with fallback to CPU
- **📊 Confidence Scores** - Top-5 predictions with probability percentages
- **🔄 Flexible Input** - Support for both file uploads and base64 encoded images
- **🛡️ CORS Enabled** - Ready for integration with frontend applications
- **📱 Mobile Friendly** - Works seamlessly with web and mobile clients
- **🚀 Production Ready** - Includes health checks and comprehensive error handling
- **☁️ Cloud Deployable** - Easy deployment with ngrok, Heroku, AWS, or Docker

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- 2GB+ free disk space (for model downloads)
- GPU optional but recommended (NVIDIA CUDA compatible GPU for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crop-recognition-api.git
   cd crop-recognition-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   python app.py
   ```

   You should see:
   ```
   ============================================================
   🌾 CROP RECOGNITION API SERVER
   ============================================================
   ✅ Server starting on http://localhost:5000
   ✅ Health check: http://localhost:5000/health
   ✅ Main endpoint: http://localhost:5000/recognize
   ============================================================
   ```

5. **Test the API**
   ```bash
   curl http://localhost:5000/health
   ```

---

## 📖 API Documentation

### Base URL
```
http://localhost:5000  (local)
https://your-ngrok-url  (ngrok tunnel)
```

### Endpoints

#### 1. **POST** `/recognize` - Upload Image File
Recognize crops from an uploaded image file.

**Request:**
```bash
curl -X POST http://localhost:5000/recognize \
  -F "image=@path/to/crop.jpg"
```

**Request Body:**
- `image` (file, required) - Image file (JPEG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "primary_crop": "tomato",
  "confidence": 94.53,
  "all_predictions": [
    {
      "crop_name": "tomato",
      "confidence": 94.53
    },
    {
      "crop_name": "pepper",
      "confidence": 3.21
    },
    {
      "crop_name": "eggplant",
      "confidence": 1.89
    }
  ],
  "model": "wambugu71/crop_leaf_diseases_vit"
}
```

---

#### 2. **POST** `/recognize-base64` - Send Base64 Image
Recognize crops from a base64 encoded image string.

**Request:**
```bash
curl -X POST http://localhost:5000/recognize-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
  }'
```

**Request Body:**
```json
{
  "image": "base64_encoded_string_or_data_url"
}
```

**Response:** Same as `/recognize`

---

#### 3. **GET** `/health` - Health Check
Check API status and model information.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "wambugu71/crop_leaf_diseases_vit",
  "device": "cuda"
}
```

---

#### 4. **GET** `/` - API Information
Returns a formatted HTML page with API documentation.

---

## 🔧 Configuration

### Environment Variables
```bash
PORT=5000  # Server port (default: 5000)
```

### Model Selection
Edit the model name in `app.py`:
```python
recognizer = CropRecognizer(model_name="wambugu71/crop_leaf_diseases_vit")
```

**Available Models:**
- `wambugu71/crop_leaf_diseases_vit` (Recommended - Crop & Leaf Disease Detection)
- `google/vit-base-patch16-224` (Fallback - General Image Classification)

---

## 🌐 Deployment

### Option 1: ngrok (Quick Testing)

1. **Install ngrok**
   ```bash
   # Download from https://ngrok.com/download
   # Or: brew install ngrok (macOS)
   ```

2. **Run your Flask server**
   ```bash
   python app.py
   ```

3. **In another terminal, start ngrok**
   ```bash
   ngrok http 5000
   ```

4. **Use the provided HTTPS URL** in your website
   ```
   https://your-random-id.ngrok.io
   ```

---

### Option 2: Heroku Deployment

1. **Create `Procfile`**
   ```
   web: gunicorn app:app
   ```

2. **Create `requirements.txt`**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Deploy**
   ```bash
   heroku login
   git init
   git add .
   git commit -m "Initial commit"
   heroku create your-app-name
   git push heroku main
   ```

---

### Option 3: Docker

1. **Create `Dockerfile`**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t crop-recognition .
   docker run -p 5000:5000 crop-recognition
   ```

---

### Option 4: AWS/Google Cloud/Azure
Deploy as a containerized service on any cloud platform. The Flask app is WSGI-compatible and works with any Python hosting provider.

---

## 💻 Frontend Integration

### JavaScript Example
```javascript
const recognizeCrop = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  try {
    const response = await fetch('https://your-api-url/recognize', {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    console.log('Identified crop:', result.primary_crop);
    console.log('Confidence:', result.confidence + '%');
    console.log('All predictions:', result.all_predictions);
  } catch (error) {
    console.error('Error:', error);
  }
};
```

### React Example
```javascript
const [predictions, setPredictions] = useState(null);

const handleImageUpload = async (e) => {
  const file = e.target.files[0];
  const formData = new FormData();
  formData.append('image', file);
  
  const response = await fetch('https://your-api-url/recognize', {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  setPredictions(data);
};

return (
  <div>
    <input type="file" onChange={handleImageUpload} accept="image/*" />
    {predictions && (
      <div>
        <h2>🌾 {predictions.primary_crop}</h2>
        <p>Confidence: {predictions.confidence}%</p>
      </div>
    )}
  </div>
);
```

---

## 📊 Performance

| Component | Details |
|-----------|---------|
| **Model** | Vision Transformer (ViT) fine-tuned for crop recognition |
| **Input Size** | 224x224 pixels (auto-resized) |
| **Inference Time** | GPU: ~50-100ms, CPU: ~200-500ms |
| **Memory Usage** | ~2-3GB (model + inference) |
| **Supported Formats** | JPEG, PNG, BMP, GIF, WebP |
| **Max File Size** | Limited by server config (typically 16MB) |

---

## 🐛 Troubleshooting

### Issue: "Model failed to load"
**Solution:** Check your internet connection. The model will download on first run (~350MB).

### Issue: Out of memory errors
**Solution:** The API will automatically fall back to CPU mode if CUDA memory is insufficient.

### Issue: Slow predictions
**Solution:** 
- Use GPU (NVIDIA CUDA) if available
- Reduce concurrent requests
- Consider model optimization/quantization

### Issue: CORS errors on frontend
**Solution:** The API has CORS enabled by default. If issues persist, check your frontend URL matches expectations.

---

## 📦 Requirements

```
Flask==2.3.0
Flask-CORS==4.0.0
torch==2.0.0
transformers==4.30.0
Pillow==9.5.0
```

Full requirements available in `requirements.txt`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support

- **Issues:** Use [GitHub Issues](https://github.com/yourusername/crop-recognition-api/issues)
- **Discussions:** Start a [GitHub Discussion](https://github.com/yourusername/crop-recognition-api/discussions)
- **Email:** your-email@example.com

---

## 🎯 Roadmap

- [ ] Batch image processing
- [ ] Disease severity classification
- [ ] Image preprocessing options
- [ ] WebSocket support for real-time streaming
- [ ] Model fine-tuning interface
- [ ] Usage analytics dashboard
- [ ] Multi-language support

---

## 🙏 Acknowledgments

- Vision Transformer model by [Google Research](https://github.com/google-research/vision_transformer)
- Crop disease dataset by [wambugu71](https://huggingface.co/wambugu71)
- Flask and PyTorch communities

---

<div align="center">

**[⬆ Back to top](#-crop-recognition-api)**

Made with ❤️ for farmers and agriculture enthusiasts

</div>

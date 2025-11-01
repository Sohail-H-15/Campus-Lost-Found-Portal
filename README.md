# 🏫 Campus Lost & Found Portal

## 📌 Overview
The **Campus Lost & Found Portal** is a web-based application designed to help students and staff easily report and recover lost or found items within the campus.  
It acts as a central platform where users can post details of lost/found items, search, and claim ownership securely.

---

## ⚙️ Tech Stack

### 🔙 Backend
- **Flask (Python Framework)** – RESTful API for handling requests
- **SQLite** – Lightweight database for item, user, and claim management
- **Flask-Mail (Gmail SMTP)** – Email verification and claim notifications
- **Werkzeug** – Secure password hashing and validation

### 🎨 Frontend
- **Vanilla JavaScript (ES6+)** – Dynamic interactions
- **Tailwind CSS** – Modern and responsive UI styling
- **Heroicons (SVG Icons)** – For clean icons
- **Google Fonts (Inter)** – Elegant typography

---

## 🚀 Features

### 👥 User Management
- Email-based signup/login
- Secure password hashing
- Email verification with tokens
- Session-based login/logout

### 📦 Item Management
- Post lost/found items with images
- Search functionality (title & description)
- **Image-based search** with AI matching (ResNet/SIFT)
- Status tracking: *Pending*, *Approved*, *Claimed*

### 📝 Claim System
- Users can claim items with proof of ownership
- Owner can approve/deny claims
- Email notifications for claim updates

### 🎨 User Interface
- Responsive design (desktop & mobile)
- Dark/Light theme toggle
- Real-time text search
- **AI-powered image search** - Upload an image to find similar items
- Image modal preview
- Toast notifications

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Install Dependencies

#### Option A: Using pip (Recommended for CPU)
```bash
pip install -r requirements.txt
```

#### Option B: Using virtual environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** PyTorch can be large (~500MB-2GB). If you only want SIFT (OpenCV), you can:
1. Install without PyTorch: `pip install -r requirements.txt --no-deps` then manually install: `pip install Flask Flask-Mail Flask-CORS opencv-python numpy Pillow`
2. The app will automatically fall back to SIFT if PyTorch is unavailable

### 3️⃣ Run the Application
```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`

### 4️⃣ Access the Application
Open your browser and navigate to:
- Frontend: Open `index.html` directly in your browser, or
- If using a local server: `http://127.0.0.1:5000`

### 5️⃣ Test Image Matching (Optional)
If you have existing images in the database without feature vectors, you can backfill them:
```bash
# Using curl or Postman, send a POST request to:
curl -X POST http://127.0.0.1:5000/backfill_features
```

Or test image matching:
```bash
curl -X POST -F "image=@path/to/test-image.jpg" -F "threshold=0.6" http://127.0.0.1:5000/match_images

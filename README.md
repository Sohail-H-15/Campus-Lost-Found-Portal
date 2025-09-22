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
- Status tracking: *Pending*, *Approved*, *Claimed*

### 📝 Claim System
- Users can claim items with proof of ownership
- Owner can approve/deny claims
- Email notifications for claim updates

### 🎨 User Interface
- Responsive design (desktop & mobile)
- Dark/Light theme toggle
- Real-time search
- Image modal preview
- Toast notifications

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# SafeSpace AI · Mental Health Companion & Therapeutic Sanctuary 🌿

> **SafeSpace** is an empathetic, AI-powered mental health companion designed to listen without judgment, help deconstruct overwhelming thoughts using cognitive behavioral frameworks (CBT/ACT), and provide immediate grounding exercises and crisis support.

---

## ✨ Key Features & Upgrades

### 🧠 1. Multi-Turn Conversational Memory
* SafeSpace maintains context across your session rather than treating each prompt in isolation.
* Adapts naturally to your conversational flow and references previous insights during the conversation.

### ⚡ 2. Real-Time Streaming Responses (SSE)
* Powered by Server-Sent Events (`/ask/stream`), responses stream smoothly word-by-word with instant feedback instead of long, frustrating loading pauses.

### 🎙️ 3. Voice Input & Voice Readout (Accessibility)
* **Speech-to-Text:** Tap the 🎙️ microphone icon to speak naturally via voice input.
* **Text-to-Speech:** Tap the 🔊 icon on any AI response to listen to a soothing read-aloud.

### 🎧 4. Calming Ambient Soundscapes (100% Offline)
* Integrated Web Audio synthesizer generates continuous, soothing nature soundscapes:
  - 🌧️ **Gentle Rainfall**
  - 🌊 **Rhythmic Ocean Waves**
  - 🍃 **Forest Breeze**
  - 💨 **Pink Noise for Sleep & Focus**
* Requires zero external MP3 downloads — synthesizes audio directly in your browser.

### 🚨 5. Comprehensive Crisis Safety Protocol
* Multi-lingual crisis detection (English, Hindi, and Hinglish keywords).
* Dedicated **SOS 🚨** modal with one-tap calling to verified helplines:
  - **India:** Tele-MANAS (`14416`), iCall (`9152987821`), Vandrevala (`+91 9999 666 555`), National (`112`)
  - **US & Canada:** Suicide & Crisis Lifeline (`988`), Crisis Text Line (`741741`)
  - **UK:** Samaritans (`116 123`), NHS (`111`), Shout (`85258`)
  - **International:** Find A Helpline directory
* Cloud-safe emergency logger (prevents headless Linux crashes on Render).

### 🌙 6. Day & Calming Midnight Dark Mode
* Instant toggle between soothing Light Claymorphism and deep Midnight Navy theme to reduce sensory overload and eye strain during late-night anxiety or insomnia.

### 📊 7. Mood Tracker & Analytics
* Daily emotional logging (1 to 5 scale with emoji).
* Check-in **streak counter** (e.g. *🔥 5 Day Streak*).
* Average mood score calculation and 14-day interactive Chart.js line trend.
* Delete log entries as needed.

### 📁 8. Conversation Management & Export
* **Rename:** Click the ✏️ pencil icon to customize session titles.
* **Delete:** Remove individual chats or clear all conversations.
* **Export:** One-click download of your chat transcript as a Markdown/Text file to share with human therapists or for personal journaling.

### 🧘 9. 8 Interactive Grounding & Breathing Exercises
1. **4-7-8 Breathing** (Interactive expanding/contracting breath guide)
2. **Box Breathing** (4s equal box cycle used by Navy SEALs)
3. **5-4-3-2-1 Grounding** (Sensory anchor to halt panic spirals)
4. **Progressive Muscle Relaxation** (10-minute somatic release guide)
5. **Mindful Body Scan** (Systematic tension release)
6. **CBT Worry Deconstruction** (Step-by-step cognitive reframing)
7. **Cold Water Dive Reflex Reset** (Vagus nerve heart rate deceleration)
8. **Neuroplastic Gratitude Anchor** (Counteracting threat bias)

---

## 🛠️ Tech Stack & Architecture

- **Backend:** FastAPI (Python 3.10+)
- **LLM Engine:**
  - **Tier 1:** Groq Cloud (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) — sub-second latency
  - **Tier 2:** Google Gemini (`gemini-2.0-flash`, `gemini-1.5-flash`)
  - **Tier 3:** Local Ollama (`alibayram/medgemma:4b`)
  - **Tier 4:** Instant Rule-based Therapeutic Knowledge Base (0ms offline fallback)
- **Emotion Classifier:** Scikit-learn Logistic Regression + TF-IDF Vectorizer
- **Database:** SQLite (default local) or PostgreSQL (Supabase / Render Postgres via `DATABASE_URL`)
- **Security:** PBKDF2-HMAC-SHA256 password hashing & JWT Bearer token authentication
- **Frontend:** Single-page responsive HTML5/CSS3/Vanilla JS with Claymorphism & Web Audio API

---

## 🚀 Local Setup Instructions

### 1. Clone & Navigate
```bash
cd mental-health-bot-main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root:
```env
# Optional LLM API Keys (Falls back to offline companion engine if not provided)
GROQ_API_KEY="your-groq-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# Database (Leave blank to use local SQLite database)
# DATABASE_URL="postgresql://user:password@host:port/dbname"

# Security
JWT_SECRET="your-secure-jwt-secret-key"

# Local Ollama (Optional)
# OLLAMA_URL="http://localhost:11434/api/chat"
# OLLAMA_MODEL="alibayram/medgemma:4b"
```

### 4. Run the Server
```bash
python main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## ☁️ Deployment on Render

1. Create a **New Web Service** on [Render.com](https://render.com).
2. Connect your GitHub repository containing this codebase.
3. Configure the following settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. In the **Environment Variables** tab on Render, add:
   - `GROQ_API_KEY` (recommended for ultra-fast streaming responses)
   - `GEMINI_API_KEY` (optional backup)
   - `JWT_SECRET` (e.g. generate a random 32-character string)
   - `DATABASE_URL` (optional: connect a free Supabase or Render PostgreSQL database)

---

## 🛡️ Crisis & Disclaimer
SafeSpace AI is designed for emotional support, stress management, and self-reflection. It is **not** a licensed therapist, clinical doctor, or a replacement for professional healthcare or emergency services. If you or someone you know is in crisis, please use the in-app **SOS 🚨** button or dial your local emergency services (112, 988, 911, or 111).

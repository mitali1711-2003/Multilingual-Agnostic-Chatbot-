# Campus Multilingual Chatbot (Hindi · English · Marathi)

Modern Flask-based chatbot for campus reception & student queries with:

- Multilingual support (Hindi, English, Marathi)
- NLP-powered FAQ matching using sentence embeddings
- Admin dashboard to upload Kaggle CSV datasets
- Authentication with hashed passwords
- SQLite storage and cloud-ready deployment

---

## 1. Tech Stack

- **Frontend**: HTML, CSS, JavaScript (vanilla, modern UI)
- **Backend**: Python, Flask
- **Database**: SQLite
- **ML / NLP**:
  - `langdetect` for language detection
  - `sentence-transformers` (multilingual MiniLM) for semantic search
  - `numpy`, `scikit-learn` utilities
- **Deployment**: Gunicorn + any cloud (Render, Railway, Azure, etc.) or local server
- **Security**:
  - `werkzeug.security` password hashing
  - Session-based authentication

---

## 2. Project Structure

```text
language-agnostic-chatbot/
  app.py                # Flask app factory & routes
  config.py             # Configuration (Dev/Prod, DB path, secrets)
  models.py             # SQLAlchemy models (User, FAQ)
  nlp_engine.py         # Language detection & semantic FAQ retrieval
  requirements.txt      # Python dependencies
  chatbot.db            # SQLite DB (created at runtime)
  templates/
    base.html           # Shared layout (navbar, footer)
    index.html          # Chat interface (chat UI, language selector)
    login.html          # Login page
    register.html       # Registration page
    admin.html          # Admin dashboard (CSV upload)
  static/
    css/styles.css      # Modern responsive styling
    js/chat.js          # Chat interaction logic (AJAX, UI updates)
  README.md             # This documentation
```

---

## 3. Setup Instructions (VS Code / local)

**Python version:** Use **Python 3.12 or 3.13**. Python 3.14+ is not yet supported (SQLAlchemy and some dependencies will fail).

1. **Create & activate a virtual environment** (recommended):

   ```bash
   cd /path/to/language-agnostic-chatbot
   python3.12 -m venv venv   # or: python3.13 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set optional environment variables** (for security in production):

   ```bash
   export SECRET_KEY="your-strong-secret-key"
   export SECURITY_PASSWORD_SALT="another-random-salt"
   # export DATABASE_URL="sqlite:///chatbot.db"  # default already uses this
   ```

4. **Run the app (development)**:

   ```bash
   python app.py
   ```

   Then open `http://localhost:5000` in your browser.

---

## 4. Authentication & Roles

- **Register** at `/register`
  - You can tick **"Register as admin"** to create an admin user.
- **Login** at `/login`
- Once logged in:
  - `/` – main **chat interface**
  - `/admin` – **admin dashboard** (only for admins)

Under the hood:

- Passwords are stored with **strong hashing** via `werkzeug.security`.
- Flask sessions keep you logged in (using `flask-login`).

---

## 5. Using Kaggle FAQ Dataset

You can use any Kaggle dataset that has campus / reception FAQ-style Q&A, for example:

- Datasets with columns like:
  - `question` – the user question text
  - `answer` – the appropriate reply
  - `language` *(optional)* – language code (`en`, `hi`, `mr`)
  - `category` *(optional)* – topic (`Admissions`, `Hostel`, etc.)

### 5.1. Download from Kaggle

1. Go to Kaggle, search e.g. **"campus faq"**, **"college helpdesk faq"**, or **"reception frequently asked questions"**.
2. Download the dataset as **CSV**.

### 5.2. Upload into the chatbot

1. Run the app and login as an admin.
2. Go to `/admin`.
3. In **"Upload FAQ dataset (CSV)"**:
   - Select your CSV file.
   - Choose a **default language**:
     - `en` – English
     - `hi` – Hindi
     - `mr` – Marathi
   - Optionally set a **default category** (e.g. `Reception FAQ`).
4. Click **"Upload & build NLP"**.

The app will:

- Insert all rows into the `faqs` table in SQLite.
- Build **sentence embeddings** (using a multilingual MiniLM model).
- Store embeddings in the DB for **fast semantic search**.

---

## 6. How the NLP Pipeline Works (Language-Agnostic Logic)

- **Language Detection**:
  - Uses `langdetect.detect(text)` to guess language.
  - Supported languages map:
    - `en` → English
    - `hi` → Hindi
    - `mr` → Marathi
  - If detection fails, the system **falls back to English**.

- **Sentence Embeddings**:
  - Uses `SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")`.
  - This model is **multilingual** and works well for Hindi, English, and Marathi.
  - Every FAQ question gets converted into a **vector** and stored as JSON in the DB.

- **Semantic Retrieval**:
  - When a user asks a question:
    1. Encode the user query into a vector.
    2. Filter FAQs by the detected language.
    3. Compute **cosine similarity** between query vector and FAQ vectors.
    4. Return the most similar answer if similarity is above a threshold.
    5. If no good match is found, show a **polite fallback** telling the user to rephrase / contact reception.

This makes the chatbot **language-agnostic** and robust to different phrasings of questions.

---

## 7. UI & UX Highlights (Innovative Touches)

- **Modern glassmorphism-style chat card** with soft shadows and gradients.
- **Language selector** directly on the hero section so users can pick Hindi / English / Marathi before chatting.
- **Quick topic chips** (Admissions, Fees, Hostel, etc.) that prefill smart prompts.
- **Typing indicator** with animated dots.
- **Responsive design**: works nicely on mobile and desktop.
- **Admin dashboard** for no-code dataset updates:
  - Upload new CSVs from Kaggle or internal sources.
  - Auto-rebuild NLP embeddings.

You can extend this with:

- Conversation history analytics (store messages in another table).
- Feedback buttons (Helpful / Not helpful) to improve responses.
- Multi-turn context-aware responses using more advanced models, if needed.

---

## 8. Deployment (Cloud / Production)

### 8.1. Gunicorn + any cloud (Render, Railway, etc.)

1. **Create a `Procfile`** (for platforms like Heroku/Render):

   ```text
   web: gunicorn app:create_app --bind 0.0.0.0:$PORT
   ```

2. **Set environment variables** in your cloud dashboard:

   - `SECRET_KEY`
   - `SECURITY_PASSWORD_SALT`
   - Optional: `DATABASE_URL` (for Postgres in production)

3. **Deploy**:

   - Push this project to a Git repo.
   - Connect it to your hosting provider.
   - Configure build: `pip install -r requirements.txt`, then run `web` process.

### 8.2. Local college server

- Install Python + virtualenv.
- Install dependencies from `requirements.txt`.
- Run via `gunicorn` (better than plain `python app.py`):

  ```bash
  gunicorn app:create_app --bind 0.0.0.0:8000
  ```

- Use Nginx / Apache as a reverse proxy if required.

---

## 9. Direct Use in VS Code

1. Open the project folder in VS Code:  
   `File → Open Folder → language-agnostic-chatbot`.
2. Select the Python interpreter from the **venv** you created.
3. Use the **Run and Debug** panel or a simple task to run:

   ```bash
   python app.py
   ```

4. Edit templates (`index.html`, `base.html`, `admin.html`) and styles (`styles.css`) to customize branding for your campus.

You now have a **fully working, deployable, multilingual campus chatbot** using the exact stack and modules you specified.

# Multilingual-Agnostic-Chatbot-
# Multilingual-Agnostic-Chatbot-

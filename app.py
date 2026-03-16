import json
import os
import sys

# SQLAlchemy and several dependencies do not yet support Python 3.14+.
if sys.version_info >= (3, 14):
    print(
        "This project requires Python 3.12 or 3.13. Python 3.14+ is not yet supported by SQLAlchemy.\n"
        "Create a venv with: python3.12 -m venv venv   or   python3.13 -m venv venv",
        file=sys.stderr,
    )
    sys.exit(1)

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    session,
)
from flask_cors import CORS
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    current_user,
    logout_user,
)
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash

from config import DevConfig
from models import Base, User, FAQ, Conversation, Message
from nlp_engine import (
    detect_language,
    retrieve_best_answer,
    get_faq_suggestions,
    SUPPORTED_LANGS,
    build_or_update_embeddings,
)

# Fallback when no FAQ matches — localized so Hindi/Marathi users get a reply
# in their language instead of always English.
FALLBACK_BY_LANG = {
    "en": (
        "Sorry, I could not find an exact match for your question. "
        "Please try rephrasing or contact the campus reception."
    ),
    "hi": (
        "क्षमा करें, आपके प्रश्न का सटीक उत्तर नहीं मिल सका। "
        "कृपया प्रश्न को दूसरे शब्दों में पूछें या कैंपस रिसेप्शन से संपर्क करें।"
    ),
    "mr": (
        "माफ करा, तुमच्या प्रश्नाचे अचूक उत्तर सापडले नाही. "
        "कृपया प्रश्न वेगळ्या शब्दांत विचारा किंवा कॅम्पस रिसेप्शनशी संपर्क साधा."
    ),
}


def create_app(config_class=DevConfig) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, supports_credentials=True)

    # Database setup
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"], future=True)
    Base.metadata.create_all(engine)

    # Migration: add campus column if missing (multi-campus support)
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE faqs ADD COLUMN campus VARCHAR(64)"))
            conn.commit()
        except Exception:
            conn.rollback()
        try:
            conn.execute(text("ALTER TABLE conversations ADD COLUMN campus VARCHAR(64)"))
            conn.commit()
        except Exception:
            conn.rollback()

    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db_session = scoped_session(session_factory)

    # Login manager
    login_manager = LoginManager()
    login_manager.login_view = "login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        with db_session() as session:
            return session.get(User, int(user_id))

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

    def _get_or_create_conversation(db_sql_session, campus_hint=None):
        """
        Keep a single active conversation per logged-in user in the session.
        Creates one if missing. Supports multi-campus.
        """
        if not current_user.is_authenticated:
            return None

        conv_id = session.get("conversation_id")
        conv = None
        if conv_id is not None:
            conv = db_sql_session.get(Conversation, conv_id)

        if conv is None:
            conv = Conversation(user_id=current_user.id, campus=campus_hint or "Main Campus")
            db_sql_session.add(conv)
            db_sql_session.commit()
            session["conversation_id"] = conv.id

        return conv

    # -------------------- Routes: Auth --------------------

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            is_admin = request.form.get("is_admin") == "on"

            if not username or not password:
                flash("Username and password are required.", "danger")
                return redirect(url_for("register"))

            with db_session() as session:
                existing = session.query(User).filter_by(username=username).first()
                if existing:
                    flash("Username already exists.", "danger")
                    return redirect(url_for("register"))

                user = User(
                    username=username,
                    password_hash=generate_password_hash(password),
                    is_admin=is_admin,
                )
                session.add(user)
                session.commit()

            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()

            with db_session() as session:
                user = session.query(User).filter_by(username=username).first()
                if not user or not check_password_hash(user.password_hash, password):
                    flash("Invalid username or password.", "danger")
                    return redirect(url_for("login"))

            login_user(user)
            return redirect(url_for("index"))

        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    # -------------------- Routes: Core UI --------------------

    @app.route("/")
    @login_required
    def index():
        language_options = [
            {"code": code, "label": name} for code, name in SUPPORTED_LANGS.items()
        ]
        quick_categories = [
            "Admissions", "Fees", "Hostel", "Library", "Exams", "Placements",
            "Scholarships", "Transport", "Canteen", "Sports", "Computer Science",
            "MBA", "Commerce", "Science", "Arts", "Administration",
        ]
        campuses = ["Main Campus", "North Campus", "South Campus"]
        with db_session() as s:
            _get_or_create_conversation(s)
            faq_count = s.query(FAQ).count()
            try:
                campus_rows = s.query(FAQ.campus).distinct().filter(
                    FAQ.campus.isnot(None),
                    FAQ.campus != "",
                ).all()
                if campus_rows:
                    campuses = ["Main Campus"] + [r[0] for r in campus_rows if r[0]]
            except Exception:
                pass
        return render_template(
            "index.html",
            language_options=language_options,
            quick_categories=quick_categories,
            campuses=campuses,
            user=current_user,
            faq_count=faq_count,
        )

    # -------------------- Routes: Admin --------------------

    def _require_admin():
        if not current_user.is_authenticated or not current_user.is_admin:
            return False
        return True

    @app.route("/admin", methods=["GET"])
    @login_required
    def admin_dashboard():
        if not _require_admin():
            flash("Admin access required.", "danger")
            return redirect(url_for("index"))

        with db_session() as s:
            faq_count = s.query(FAQ).count()
            faqs_with_embeddings = (
                s.query(FAQ).filter(FAQ.embedding.isnot(None), FAQ.embedding != "").count()
            )

            conversation_count = s.query(Conversation).count()
            message_count = s.query(Message).count()
            avg_messages = (
                round(message_count / conversation_count, 2)
                if conversation_count > 0
                else 0
            )

            lang_distribution = (
                s.query(Message.language, func.count(Message.id))
                .group_by(Message.language)
                .all()
            )

            feedback_positive = (
                s.query(func.count(Message.id))
                .filter(Message.feedback == "positive")
                .scalar()
            )
            feedback_negative = (
                s.query(func.count(Message.id))
                .filter(Message.feedback == "negative")
                .scalar()
            )

            top_faqs = (
                s.query(FAQ.question, func.count(Message.id).label("hits"))
                .join(Message, Message.faq_id == FAQ.id)
                .group_by(FAQ.id)
                .order_by(func.count(Message.id).desc())
                .limit(5)
                .all()
            )

        return render_template(
            "admin.html",
            faq_count=faq_count,
            faqs_with_embeddings=faqs_with_embeddings,
            languages=SUPPORTED_LANGS,
             conversation_count=conversation_count,
             message_count=message_count,
             avg_messages=avg_messages,
             lang_distribution=lang_distribution,
             feedback_positive=feedback_positive,
             feedback_negative=feedback_negative,
             top_faqs=top_faqs,
        )

    @app.route("/admin/upload_faqs", methods=["POST"])
    @login_required
    def upload_faqs():
        if not _require_admin():
            return jsonify({"error": "Admin access required"}), 403

        file = request.files.get("file")
        default_lang = request.form.get("default_lang", "en")
        default_category = request.form.get("default_category", "")
        default_campus = request.form.get("default_campus", "Main Campus")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = (file.filename or "").lower()
        is_json = filename.endswith(".json")

        if is_json:
            try:
                data = json.load(file)
            except Exception as exc:
                return jsonify({"error": f"Invalid JSON: {exc}"}), 400
            faqs_list = data.get("faqs", data) if isinstance(data, dict) else data
            if not isinstance(faqs_list, list):
                return jsonify({"error": "JSON must contain a 'faqs' array"}), 400
            with db_session() as s:
                for item in faqs_list:
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    if not q or not a:
                        continue
                    lang = str(item.get("language", "en")).strip() or default_lang
                    cat = str(item.get("category", "")).strip()
                    campus = str(item.get("campus", "")).strip() or default_campus
                    faq = FAQ(question=q, answer=a, language=lang, category=cat, campus=campus)
                    s.add(faq)
                s.commit()
                build_or_update_embeddings(s)
            return jsonify({"status": "ok", "count": len(faqs_list)}), 200

        import pandas as pd

        try:
            df = pd.read_csv(file)
        except Exception as exc:
            return jsonify({"error": f"Failed to read CSV: {exc}"}), 400

        required_cols = {"question", "answer"}
        if not required_cols.issubset(df.columns):
            return jsonify({"error": "CSV must contain 'question' and 'answer' columns"}), 400

        with db_session() as session:
            for _, row in df.iterrows():
                question = str(row["question"])
                answer = str(row["answer"])
                language = str(row.get("language", "")).strip() or default_lang
                category = str(row.get("category", "")).strip() or default_category
                campus = str(row.get("campus", "")).strip() or default_campus

                faq = FAQ(
                    question=question,
                    answer=answer,
                    language=language,
                    category=category,
                    campus=campus,
                )
                session.add(faq)

            session.commit()
            build_or_update_embeddings(session)

        return jsonify({"status": "ok"}), 200

    @app.route("/admin/upload_json_faqs", methods=["POST"])
    @login_required
    def upload_json_faqs():
        if not _require_admin():
            return jsonify({"error": "Admin access required"}), 403

        file = request.files.get("file")
        default_campus = request.form.get("default_campus", "Main Campus")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        try:
            data = json.load(file)
        except Exception as exc:
            return jsonify({"error": f"Invalid JSON: {exc}"}), 400

        faqs_list = data.get("faqs", data) if isinstance(data, dict) else data
        if not isinstance(faqs_list, list):
            return jsonify({"error": "JSON must contain a 'faqs' array"}), 400

        with db_session() as s:
            for item in faqs_list:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if not q or not a:
                    continue
                lang = str(item.get("language", "en")).strip() or "en"
                cat = str(item.get("category", "")).strip()
                campus = str(item.get("campus", "")).strip() or default_campus
                faq = FAQ(question=q, answer=a, language=lang, category=cat, campus=campus)
                s.add(faq)
            s.commit()
            build_or_update_embeddings(s)

        return jsonify({"status": "ok", "count": len(faqs_list)}), 200

    @app.route("/admin/load_default_dataset", methods=["POST"])
    @login_required
    def load_default_dataset():
        if not _require_admin():
            return jsonify({"error": "Admin access required"}), 403

        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "campus_multilingual_dataset.json",
        )
        if not os.path.isfile(default_path):
            return jsonify({"error": "Default dataset not found"}), 404

        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        faqs_list = data.get("faqs", [])
        default_campus = request.form.get("default_campus", "Main Campus")

        with db_session() as s:
            added = 0
            for item in faqs_list:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if not q or not a:
                    continue
                lang = str(item.get("language", "en")).strip() or "en"
                cat = str(item.get("category", "")).strip()
                # Skip duplicates
                exists = s.query(FAQ).filter_by(
                    question=q, language=lang, category=cat,
                ).first()
                if exists:
                    continue
                faq = FAQ(question=q, answer=a, language=lang, category=cat, campus=default_campus)
                s.add(faq)
                added += 1
            s.commit()
            build_or_update_embeddings(s)

        return jsonify({"status": "ok", "count": added}), 200

    @app.route("/admin/rebuild_embeddings", methods=["POST"])
    @login_required
    def rebuild_embeddings():
        if not _require_admin():
            return jsonify({"error": "Admin access required"}), 403
        with db_session() as s:
            build_or_update_embeddings(s)
        return jsonify({"status": "ok"}), 200

    # -------------------- API: Chat --------------------

    @app.route("/api/suggest", methods=["GET"])
    @login_required
    def suggest_api():
        prefix = request.args.get("q", "").strip()
        lang = request.args.get("lang") or "en"
        campus = request.args.get("campus") or "Main Campus"
        limit = min(int(request.args.get("limit", 8)), 20)

        with db_session() as s:
            suggestions = get_faq_suggestions(
                session=s,
                prefix=prefix,
                lang_hint=lang,
                campus_hint=campus,
                limit=limit,
            )
        return jsonify({"suggestions": suggestions})

    @app.route("/api/chat", methods=["POST"])
    @login_required
    def chat_api():
        data = request.get_json(force=True)
        message = (data.get("message") or "").strip()
        explicit_lang = data.get("language") or None
        campus_hint = data.get("campus") or "Main Campus"

        if not message:
            return jsonify({"error": "Empty message"}), 400

        with db_session() as s:
            conversation = _get_or_create_conversation(s)

            # Use detected language of the message for retrieval so the answer is in the
            # same language as the question. (Dropdown is for UI only; we no longer use it
            # for FAQ search, which fixed wrong-language answers e.g. English questions
            # returning Hindi replies when Hindi was selected.)
            user_lang = detect_language(message)

            # Log user message
            user_msg = Message(
                conversation_id=conversation.id,
                sender="user",
                text=message,
                language=user_lang,
            )
            s.add(user_msg)
            s.flush()

            # Use only the current message for retrieval. Previous "multi-turn" context
            # caused wrong answers (e.g. admission/hostel/canteen questions returning
            # sports or hostel answers when earlier messages were about other topics).
            best_faq, ranked = retrieve_best_answer(
                session=s,
                user_query=message,
                lang_hint=user_lang,
                campus_hint=campus_hint,
            )

            if not best_faq:
                detected = user_lang
                fallback_text = FALLBACK_BY_LANG.get(detected, FALLBACK_BY_LANG["en"])

                bot_msg = Message(
                    conversation_id=conversation.id,
                    sender="bot",
                    text=fallback_text,
                    language=detected,
                )
                s.add(bot_msg)
                s.commit()

                return jsonify(
                    {
                        "answer": fallback_text,
                        "detected_language": detected,
                        "matches": [],
                        "conversation_id": conversation.id,
                        "message_id": bot_msg.id,
                    }
                )

            matches_payload = []
            best_score = None
            for faq, score in ranked:
                if best_score is None:
                    best_score = score
                matches_payload.append(
                    {
                        "question": faq.question,
                        "answer": faq.answer,
                        "score": score,
                        "language": faq.language,
                        "category": faq.category,
                    }
                )

            bot_msg = Message(
                conversation_id=conversation.id,
                sender="bot",
                text=best_faq.answer,
                language=best_faq.language,
                faq_id=best_faq.id,
                similarity=best_score,
            )
            s.add(bot_msg)
            s.commit()

            return jsonify(
                {
                    "answer": best_faq.answer,
                    "detected_language": best_faq.language,
                    "matches": matches_payload,
                    "conversation_id": conversation.id,
                    "message_id": bot_msg.id,
                }
            )

    @app.route("/api/feedback", methods=["POST"])
    @login_required
    def feedback_api():
        data = request.get_json(force=True)
        message_id = data.get("message_id")
        value = data.get("value")

        if not message_id or value not in ("up", "down"):
            return jsonify({"error": "Invalid feedback payload"}), 400

        with db_session() as s:
            msg = s.get(Message, int(message_id))
            if not msg or msg.sender != "bot":
                return jsonify({"error": "Feedback can only be left on bot messages"}), 400

            msg.feedback = "positive" if value == "up" else "negative"
            s.commit()

        return jsonify({"status": "ok"})

    # Simple health endpoint
    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)


#!/usr/bin/env python3
"""
Load the default campus FAQ dataset into the database.
Run this once to enable the chatbot, or use the Admin panel in the web app.

Usage:
    python load_dataset.py
"""
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DevConfig
from models import Base, FAQ
from nlp_engine import build_or_update_embeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session


def main():
    config = DevConfig()
    engine = create_engine(config.SQLALCHEMY_DATABASE_URI, future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db_session = scoped_session(session_factory)

    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "campus_multilingual_dataset.json",
    )
    if not os.path.isfile(default_path):
        print("Error: Default dataset not found at", default_path)
        sys.exit(1)

    with open(default_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    faqs_list = data.get("faqs", [])
    default_campus = "Main Campus"

    with db_session() as s:
        count = 0
        for item in faqs_list:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a:
                continue
            lang = str(item.get("language", "en")).strip() or "en"
            cat = str(item.get("category", "")).strip()
            faq = FAQ(
                question=q,
                answer=a,
                language=lang,
                category=cat,
                campus=default_campus,
            )
            s.add(faq)
            count += 1
        s.commit()
        print(f"Loaded {count} FAQs into database.")
        print("Building embeddings (this may take 1-2 minutes)...")
        build_or_update_embeddings(s)
        print("Done. Embeddings built. The chatbot is ready to use.")


if __name__ == "__main__":
    main()

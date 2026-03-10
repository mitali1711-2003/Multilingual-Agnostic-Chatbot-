import json
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, or_, func
from sqlalchemy.orm import Session

from models import FAQ


SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

# Keywords to route queries to the correct category (avoids wrong matches)
CATEGORY_KEYWORDS = {
    "Admissions": ["admission", "admit", "apply", "eligibility", "documents", "प्रवेश", "एडमिशन"],
    "Placements": ["placement", "placements", "companies", "package", "recruitment", "प्लेसमेंट", "नौकरी"],
    "Hostel": ["hostel", "hostels", "lodging", "हॉस्टल", "हॉस्टेल"],
    "Library": ["library", "books", "issued", "lib", "पुस्तकालय"],
    "Exams": ["exam", "examination", "revaluation", "semester", "परीक्षा"],
    "Scholarships": ["scholarship", "scholarships", "financial aid", "छात्रवृत्ति"],
    "Transport": ["bus", "transport", "travel", "बस", "यातायात"],
    "Canteen": ["canteen", "food", "mess", "खाना", "कॅन्टीन"],
    "Sports": ["sports", "sport", "gym", "खेल"],
    "Computer Science": ["computer", "programming", "coding", "cs", "it department"],
    "MBA": ["mba", "business", "management"],
    "Arts": ["arts", "humanities", "कला"],
    "Academics": ["attendance", "academic", "curriculum", "सिलेबस"],
    "Facilities": ["wifi", "wi-fi", "lab", "facility", "सुविधा"],
    "Administration": ["grievance", "complaint", "admin", "office", "शिकायत"],
}


def _detect_category_hint(query: str) -> Optional[str]:
    """
    Return the most likely category for the query, or None.

    Instead of returning the first category that matches a keyword (which can
    bias towards generic categories like Admissions), we:
    - Count how many keywords from each category appear in the query.
    - Prefer the category with the highest match count.
    - In case of ties, prefer more "specific" categories (e.g. Hostel) over
      generic ones (e.g. Admissions).

    This fixes cases like "hostel admission process", which previously matched
    Admissions first and therefore returned generic admission FAQs instead of
    hostel-related information.
    """
    q = query.lower().strip()
    if not q:
        return None

    # Categories ordered from more specific to more generic for tie‑breaking.
    specificity_order = [
        "Hostel",
        "Library",
        "Placements",
        "Sports",
        "Canteen",
        "Transport",
        "Scholarships",
        "Computer Science",
        "MBA",
        "Arts",
        "Academics",
        "Facilities",
        "Administration",
        "Admissions",
    ]
    specificity_rank = {cat: idx for idx, cat in enumerate(specificity_order)}

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = 0
        for kw in keywords:
            if kw.lower() in q:
                count += 1
        if count > 0:
            scores[category] = count

    if not scores:
        return None

    # Pick category with highest score; break ties using specificity_order.
    best_category = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            specificity_rank.get(item[0], len(specificity_order)),
        ),
    )[0][0]

    return best_category


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Multilingual MiniLM model that supports Hindi, English, Marathi
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def detect_language(text: str) -> str:
    """
    Detect language code using langdetect.
    Falls back to English if detection fails.
    """
    try:
        lang = detect(text)
        if lang in SUPPORTED_LANGS:
            return lang
    except Exception:
        pass
    return "en"


def encode_sentences(sentences: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def _parse_embedding(embedding_str: Optional[str]) -> Optional[np.ndarray]:
    if not embedding_str:
        return None
    try:
        arr = np.array(json.loads(embedding_str), dtype="float32")
        return arr
    except Exception:
        return None


def build_or_update_embeddings(session: Session) -> None:
    """
    Ensure all FAQ rows have embeddings. Called from admin panel
    after Kaggle CSV upload or FAQ changes.
    """
    faqs = session.execute(select(FAQ)).scalars().all()
    texts_to_encode = []
    faq_ids = []

    for faq in faqs:
        if not faq.embedding:
            texts_to_encode.append(faq.question)
            faq_ids.append(faq.id)

    if not texts_to_encode:
        return

    embeddings = encode_sentences(texts_to_encode)

    for faq_id, emb in zip(faq_ids, embeddings):
        faq = session.get(FAQ, faq_id)
        if faq:
            faq.embedding = json.dumps(emb.tolist())

    session.commit()


def retrieve_best_answer(
    session: Session,
    user_query: str,
    lang_hint: Optional[str] = None,
    campus_hint: Optional[str] = None,
    top_k: int = 3,
) -> Tuple[Optional[FAQ], List[Tuple[FAQ, float]]]:
    """
    Retrieve the most similar FAQ answer using cosine similarity over embeddings.
    Supports multi-campus filtering. Returns best FAQ and list of (FAQ, score).
    """
    detected_lang = lang_hint or detect_language(user_query)
    category_hint = _detect_category_hint(user_query)

    base_filter = FAQ.language == detected_lang
    if campus_hint:
        base_filter = base_filter & or_(
            FAQ.campus == campus_hint,
            FAQ.campus.is_(None),
            FAQ.campus == "",
        )
    # Match category case-insensitively so "Admissions" / "admissions" both work
    if category_hint:
        base_filter = base_filter & (func.lower(FAQ.category) == category_hint.lower())

    faqs = (
        session.execute(select(FAQ).where(base_filter))
        .scalars()
        .all()
    )

    # When the user clearly asked about a topic (category_hint), do NOT fall back
    # to other categories. Otherwise e.g. "Tell me about Admissions" can match
    # "Tell me about hostel at the campus" and return the wrong answer.
    if not faqs and category_hint:
        return None, []

    if not faqs:
        faqs = session.execute(select(FAQ)).scalars().all()
        if not faqs:
            return None, []

    faq_embeddings = []
    valid_faqs = []
    for faq in faqs:
        emb = _parse_embedding(faq.embedding)
        if emb is not None:
            faq_embeddings.append(emb)
            valid_faqs.append(faq)

    if not valid_faqs:
        return None, []

    query_emb = encode_sentences([user_query])[0]
    matrix = np.vstack(faq_embeddings)  # shape (N, D)
    scores = np.dot(matrix, query_emb)  # cosine because embeddings are normalized

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    ranked = [(valid_faqs[i], float(scores[i])) for i in ranked_indices]

    best_faq, best_score = ranked[0]
    # Threshold tuned for natural phrasing ("Tell me about Placements/Hostel at the campus")
    if best_score < 0.25:
        return None, ranked

    return best_faq, ranked


def get_faq_suggestions(
    session: Session,
    prefix: str = "",
    lang_hint: Optional[str] = None,
    campus_hint: Optional[str] = None,
    limit: int = 8,
) -> List[dict]:
    """
    Return FAQ questions as suggestions (before/during typing).
    When prefix is empty, returns popular/sample questions.
    When prefix given, filters by substring match.
    """
    base = select(FAQ).where(FAQ.question != "")
    if lang_hint:
        base = base.where(FAQ.language == lang_hint)
    if campus_hint and campus_hint.strip():
        base = base.where(
            or_(
                FAQ.campus == campus_hint,
                FAQ.campus.is_(None),
                FAQ.campus == "",
            )
        )

    if prefix and prefix.strip():
        like_pattern = f"%{prefix.strip()}%"
        base = base.where(FAQ.question.like(like_pattern))
    base = base.limit(limit)
    faqs = session.execute(base).scalars().all()

    return [
        {"question": f.question, "answer": f.answer, "category": f.category}
        for f in faqs
    ]


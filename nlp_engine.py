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
# Includes English, Hindi, and Marathi keywords for each category.
CATEGORY_KEYWORDS = {
    "Admissions": [
        "admission", "admit", "apply", "eligibility", "documents",
        "प्रवेश", "एडमिशन", "पात्रता", "दस्तावेज़", "कागदपत्रे",
    ],
    "Placements": [
        "placement", "placements", "companies", "package", "recruitment",
        "प्लेसमेंट", "नौकरी", "कंपनी", "कंपन्या", "भरती", "पॅकेज",
    ],
    "Hostel": [
        "hostel", "hostels", "lodging",
        "हॉस्टल", "हॉस्टेल", "छात्रावास",
    ],
    "Library": [
        "library", "books", "issued", "lib",
        "पुस्तकालय", "किताब", "किताबें", "ग्रंथालय", "पुस्तके",
    ],
    "Exams": [
        "exam", "examination", "revaluation", "semester",
        "परीक्षा", "पुनर्मूल्यांकन", "सेमेस्टर",
    ],
    "Scholarships": [
        "scholarship", "scholarships", "financial aid",
        "छात्रवृत्ति", "शिष्यवृत्ती", "स्कॉलरशिप",
    ],
    "Transport": [
        "bus", "transport", "travel",
        "बस", "यातायात", "वाहतूक", "परिवहन",
    ],
    "Canteen": [
        "canteen", "food", "mess",
        "खाना", "कॅन्टीन", "कैंटीन", "भोजन", "जेवण",
    ],
    "Sports": [
        "sports", "sport", "gym",
        "खेल", "क्रीडा", "व्यायामशाला", "जिम",
    ],
    "Computer Science": [
        "computer", "programming", "coding", "cs", "it department",
        "कंप्यूटर", "कॉम्प्युटर", "प्रोग्रामिंग",
    ],
    "MBA": [
        "mba", "business", "management",
        "व्यवसाय", "प्रबंधन", "व्यवस्थापन",
    ],
    "Commerce": [
        "commerce", "accounting", "ca ", "taxation",
        "कॉमर्स", "वाणिज्य", "लेखांकन",
    ],
    "Science": [
        "science", "physics", "chemistry", "biology",
        "विज्ञान", "भौतिकी", "रसायन",
    ],
    "Arts": [
        "arts", "humanities",
        "कला", "मानविकी",
    ],
    "Academics": [
        "attendance", "academic", "curriculum", "syllabus",
        "उपस्थिति", "सिलेबस", "अभ्यासक्रम", "पाठ्यक्रम",
    ],
    "Facilities": [
        "wifi", "wi-fi", "lab", "facility",
        "सुविधा", "लैब", "लॅब", "वायफाय",
    ],
    "Administration": [
        "grievance", "complaint", "admin", "office",
        "शिकायत", "तक्रार", "कार्यालय",
    ],
    "Fees": [
        "fee", "fees", "payment", "installment",
        "शुल्क", "फीस", "किस्त", "हप्ता",
    ],
}


def _detect_category_hint(query: str) -> Optional[str]:
    """
    Return the most likely category for the query, or None.

    Counts keyword matches per category and picks the highest scorer.
    Ties are broken by specificity (more specific categories win).
    """
    q = query.lower().strip()
    if not q:
        return None

    specificity_order = [
        "Hostel", "Library", "Placements", "Sports", "Canteen",
        "Transport", "Scholarships", "Computer Science", "MBA",
        "Commerce", "Science", "Arts", "Academics", "Facilities",
        "Administration", "Fees", "Admissions",
    ]
    specificity_rank = {cat: idx for idx, cat in enumerate(specificity_order)}

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in q)
        if count > 0:
            scores[category] = count

    if not scores:
        return None

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
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _has_devanagari(text: str) -> bool:
    """True if the text contains Devanagari script (Hindi/Marathi)."""
    return any("\u0900" <= c <= "\u097F" for c in text)


# Common Romanized Hindi/Hinglish words
_HINGLISH_WORDS = frozenset(
    "kya kaise kab kahan kyu ki ke ka ko se mein me par aur bhi hai hain ho "
    "batao bataiye bataen bataye sampark karein mil chahiye kitna batayein "
    "mujhe hamein iski uski yeh wo hota hoti".split()
)


def _looks_like_hinglish(text: str) -> bool:
    """True if the text looks like Hinglish (Roman script but Hindi-style words)."""
    t = text.lower().strip()
    if not t or _has_devanagari(text):
        return False
    words = set(t.replace("?", " ").replace(".", " ").split())
    matches = sum(1 for w in words if w in _HINGLISH_WORDS)
    return matches >= 2 or (matches >= 1 and len(words) <= 6)


def detect_language(text: str) -> str:
    """
    Detect language code.
    - Devanagari text -> Hindi (or Marathi if detected).
    - Hinglish (Roman + Hindi words like kya, hai) -> Hindi.
    """
    try:
        lang = detect(text)
        if lang in SUPPORTED_LANGS:
            return lang
        if lang == "en" and _has_devanagari(text):
            return "hi"
        if lang == "en" and _looks_like_hinglish(text):
            return "hi"
    except Exception:
        pass
    if _has_devanagari(text):
        return "hi"
    if _looks_like_hinglish(text):
        return "hi"
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
    after CSV/JSON upload or FAQ changes.
    """
    faqs = session.execute(select(FAQ)).scalars().all()
    texts_to_encode = []
    faq_ids = []

    for faq in faqs:
        if not faq.embedding:
            # Include answer prefix for richer semantic signal
            text = faq.question
            if faq.answer:
                text = f"{faq.question} {faq.answer[:100]}"
            texts_to_encode.append(text)
            faq_ids.append(faq.id)

    if not texts_to_encode:
        return

    embeddings = encode_sentences(texts_to_encode)

    for faq_id, emb in zip(faq_ids, embeddings):
        faq = session.get(FAQ, faq_id)
        if faq:
            faq.embedding = json.dumps(emb.tolist())

    session.commit()


def _rank_faqs(faqs: List[FAQ], query_emb: np.ndarray, top_k: int) -> List[Tuple[FAQ, float]]:
    """Rank FAQs by cosine similarity to query embedding."""
    faq_embeddings = []
    valid_faqs = []
    for faq in faqs:
        emb = _parse_embedding(faq.embedding)
        if emb is not None:
            faq_embeddings.append(emb)
            valid_faqs.append(faq)

    if not valid_faqs:
        return []

    matrix = np.vstack(faq_embeddings)
    scores = np.dot(matrix, query_emb)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(valid_faqs[i], float(scores[i])) for i in ranked_indices]


def retrieve_best_answer(
    session: Session,
    user_query: str,
    lang_hint: Optional[str] = None,
    campus_hint: Optional[str] = None,
    top_k: int = 3,
) -> Tuple[Optional[FAQ], List[Tuple[FAQ, float]]]:
    """
    Retrieve the best FAQ answer using a two-phase strategy:

    Phase 1: Search FAQs matching the detected language + category + campus.
    Phase 2 (cross-lingual fallback): If Phase 1 finds nothing good, search
             ALL languages for the same category, leveraging the multilingual
             embedding model's cross-lingual similarity.

    Returns (best_faq, ranked_list) or (None, ranked_list).
    """
    detected_lang = lang_hint or detect_language(user_query)
    category_hint = _detect_category_hint(user_query)
    query_emb = encode_sentences([user_query])[0]

    def _build_campus_filter():
        if campus_hint:
            return or_(
                FAQ.campus == campus_hint,
                FAQ.campus.is_(None),
                FAQ.campus == "",
            )
        return None

    def _fetch_faqs(lang_filter=None, category_filter=None):
        filters = []
        if lang_filter is not None:
            filters.append(FAQ.language == lang_filter)
        campus_f = _build_campus_filter()
        if campus_f is not None:
            filters.append(campus_f)
        if category_filter:
            filters.append(func.lower(FAQ.category) == category_filter.lower())

        stmt = select(FAQ)
        for f in filters:
            stmt = stmt.where(f)
        return session.execute(stmt).scalars().all()

    # --- Phase 1: Same language + category ---
    faqs = _fetch_faqs(lang_filter=detected_lang, category_filter=category_hint)
    if faqs:
        ranked = _rank_faqs(faqs, query_emb, top_k)
        if ranked and ranked[0][1] >= 0.25:
            return ranked[0][0], ranked

    # --- Phase 2: Cross-lingual fallback (all languages, same category) ---
    if category_hint:
        faqs_cross = _fetch_faqs(lang_filter=None, category_filter=category_hint)
        if faqs_cross:
            ranked = _rank_faqs(faqs_cross, query_emb, top_k)
            if ranked and ranked[0][1] >= 0.25:
                return ranked[0][0], ranked

    # --- Phase 3: Same language, no category filter ---
    faqs_no_cat = _fetch_faqs(lang_filter=detected_lang, category_filter=None)
    if faqs_no_cat:
        ranked = _rank_faqs(faqs_no_cat, query_emb, top_k)
        if ranked and ranked[0][1] >= 0.3:
            return ranked[0][0], ranked

    # --- Phase 4: All languages, no category filter (full cross-lingual) ---
    faqs_all = _fetch_faqs(lang_filter=None, category_filter=None)
    if faqs_all:
        ranked = _rank_faqs(faqs_all, query_emb, top_k)
        if ranked and ranked[0][1] >= 0.35:
            return ranked[0][0], ranked
        return None, ranked

    return None, []


def get_faq_suggestions(
    session: Session,
    prefix: str = "",
    lang_hint: Optional[str] = None,
    campus_hint: Optional[str] = None,
    limit: int = 8,
) -> List[dict]:
    """
    Return FAQ questions as suggestions (before/during typing).
    When prefix is empty, returns sample questions.
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

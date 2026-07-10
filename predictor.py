import sqlite3
import os
import re
import json
import math
from collections import Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database", "learn.db")

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "as", "until", "while", "it", "its", "what",
    "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "they", "them", "their", "please", "tell",
    "show", "list", "find", "get", "give", "need", "want", "like",
    "analyze", "analyse", "across", "between", "both", "each",
    "also", "then", "than", "very", "just", "please", "can",
}


def _tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def _jaccard_similarity(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def _tfidf_vectors(all_token_lists):
    doc_freq = Counter()
    for tokens in all_token_lists:
        for t in set(tokens):
            doc_freq[t] += 1
    n_docs = len(all_token_lists)

    vocab = sorted(doc_freq.keys())
    word_index = {w: i for i, w in enumerate(vocab)}

    vectors = []
    for tokens in all_token_lists:
        vec = [0.0] * len(vocab)
        term_freq = Counter(tokens)
        for word, tf in term_freq.items():
            if word in word_index:
                idf = math.log((n_docs + 1) / (doc_freq[word] + 1)) + 1
                vec[word_index[word]] = tf * idf
        vectors.append(vec)
    return vectors


def _cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if not norm1 or not norm2:
        return 0.0
    return dot / (norm1 * norm2)


def _normalize_question(question):
    q = question.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def _get_all_successful_queries(limit=500):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, sql_generated, answer FROM query_log WHERE success = 1 AND answer != '' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def find_similar_question(question, threshold=0.65, top_n=3):
    rows = _get_all_successful_queries()
    if not rows:
        return None

    q_normalized = _normalize_question(question)
    q_tokens = _tokenize(q_normalized)

    if not q_tokens:
        return None

    cached_questions = [r[0] for r in rows]
    all_token_lists = [q_tokens] + [_tokenize(_normalize_question(r)) for r in cached_questions]

    vectors = _tfidf_vectors(all_token_lists)
    query_vec = vectors[0]

    scored = []
    for i, (r, cached_q) in enumerate(zip(rows, cached_questions)):
        cached_tokens = _tokenize(_normalize_question(cached_q))
        if not cached_tokens:
            continue
        jaccard = _jaccard_similarity(q_tokens, cached_tokens)
        cosine = _cosine_similarity(query_vec, vectors[i + 1])
        combined = 0.4 * jaccard + 0.6 * cosine
        scored.append((combined, r[0], r[1], r[2]))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < threshold:
        return None

    return {
        "question": scored[0][1],
        "sql": scored[0][2],
        "answer": scored[0][3],
        "score": round(scored[0][0], 3),
    }


def export_similarity_report(limit=200):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, COUNT(*) as cnt FROM query_log WHERE success = 1 GROUP BY question HAVING cnt > 1 ORDER BY cnt DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"question": r[0], "frequency": r[1]} for r in rows]


def get_cache_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM query_log WHERE success = 1").fetchone()[0]
    unique_questions = conn.execute("SELECT COUNT(DISTINCT question) FROM query_log WHERE success = 1").fetchone()[0]
    repeated = conn.execute(
        "SELECT COUNT(*) FROM (SELECT question FROM query_log WHERE success = 1 GROUP BY question HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    conn.close()
    return {
        "total_cached_answers": total,
        "unique_questions": unique_questions,
        "repeated_questions": repeated,
    }

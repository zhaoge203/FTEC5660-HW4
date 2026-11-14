"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


# Populated from document metadata by chunk_document; no corpus-specific names are stored here.
_CORPUS_COMPANIES: dict[str, str] = {}
_QUERY_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "held", "how", "in", "is", "it", "its", "much", "of", "on", "or",
    "report", "reported", "set", "spend", "the", "their", "to", "was", "were", "what", "when",
    "which", "who", "whose", "with", "year", "fy", "usd", "million", "millions", "percent",
    "december", "q4", "fourth", "quarter", "end", "ended", "shares", "share", "company",
    "higher", "more", "generate", "generated", "earn", "earned", "give", "gives", "value",
    "greater", "greatest", "larger", "largest", "lower", "lowest", "less", "least", "fewer", "fewest",
    "highest", "most", "number", "amount", "where", "located",
}
_CONCEPTS = (
    (("cash", "equivalents"), ("cash", "equivalents")),
    (("cash",), ("cash", "equivalents")),
    (("total", "assets"), ("total", "assets")),
    (("net", "income"), ("net", "income")),
    (("operating", "income"), ("operating", "income")),
    (("operating", "profit"), ("operating", "income")),
    (("gross", "profit"), ("gross", "profit")),
    (("accounts", "receivable"), ("accounts", "receivable", "net")),
    (("receivables",), ("accounts", "receivable", "net")),
    (("long", "term", "borrowing"), ("long", "term", "borrowings")),
    (("long", "term", "debt"), ("long", "term", "borrowings")),
    (("borrowings",), ("borrowings",)),
    (("operating", "activities"), ("net", "cash", "operating", "activities")),
    (("inventories",), ("inventories",)),
    (("total", "equity"), ("total", "equity")),
    (("employees",), ("employees", "total")),
    (("headcount",), ("employees", "total")),
    (("staff",), ("employees", "total")),
    (("workforce",), ("employees", "total")),
    (("chief", "financial"), ("chief", "financial", "officer")),
    (("cfo",), ("chief", "financial", "officer")),
    (("chief", "executive"), ("chief", "executive", "officer")),
    (("ceo",), ("chief", "executive", "officer")),
    (("audit", "firm"), ("audited", "financial", "statements")),
    (("auditor",), ("audited", "financial", "statements")),
    (("headquartered",), ("headquarters",)),
    (("headquarters",), ("headquarters",)),
    (("head", "office"), ("headquarters",)),
    (("based",), ("headquarters",)),
    (("located",), ("headquarters",)),
    (("target", "price"), ("target", "price")),
    (("investment", "rating"), ("rating",)),
    (("rating",), ("rating",)),
    (("repurchase",), ("share", "repurchase", "programme")),
    (("buyback",), ("share", "repurchase", "programme")),
    (("revenue", "growth"), ("revenue", "growth")),
    (("sales", "growth"), ("revenue", "growth")),
    (("revenue",), ("revenue",)),
    (("sales",), ("revenue",)),
    (("turnover",), ("revenue",)),
)
_LEGAL_WORDS = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def _clean_text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _company_key_local(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", name.casefold())
    return " ".join(word for word in words if word not in _LEGAL_WORDS)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold().replace(",", ""))


def _query_concepts(question: str) -> list[tuple[str, ...]]:
    words = _tokens(question)
    return [output for triggers, output in _CONCEPTS if all(word in words for word in triggers)]


def _strip_known_companies(question: str) -> str:
    result = question
    for key, name in sorted(_CORPUS_COMPANIES.items(), key=lambda item: -len(item[0])):
        result = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?:\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC))?(?![A-Za-z0-9])",
                        " ", result, flags=re.IGNORECASE)
    return result


def _unknown_company_candidates(question: str) -> list[str]:
    """Extract capitalised company-like spans only from grammatical company positions."""
    name = r"([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5})"
    patterns = [
        rf"{name}(?:[’']s|[’'](?=\s))",
        rf"\bfor\s+{name}(?=\s*[,?.]|\s+in\s+USD|\s+in\s+its\b|$)",
        rf"\bgive\s+{name}(?=\s+in\s+its\b|$)",
        rf"\bdid\s+{name}(?=\s+(?:hold|have|report|generate|earn)\b)",
        rf"\bof\s+{name}(?=\s+(?:in|at|for|during)\b|\s*[,?.]|$)",
        rf"\bat\s+{name}(?=\s+(?:in|at|for|during)\b|\s*[,?.]|$)",
        rf"\b[Ww]hich\s+of\s+{name}\s+and\s+{name}(?=\s*[,?.]|\s+(?:had|has|held|reported|earned)\b|$)",
        rf":\s*{name}\s+or\s+{name}(?=\s*[?.]|$)",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            for candidate in match.groups():
                candidate = candidate.strip(" ,.:;?")
                if candidate and candidate not in found:
                    found.append(candidate)
    return found


def _is_company_candidate(candidate: str) -> bool:
    lowered = candidate.casefold()
    if any(word in lowered for word in ("securities", "equity research", "capital", "llp", "audit")):
        return False
    if re.match(r"^(?:FY|Q[1-4]|USD|Chief|Asia|Europe|Americas|December|January|February|March)\b", candidate):
        return False
    return len(_tokens(candidate)) >= 2


def _split_narrative(text: str, limit: int = 1150) -> list[str]:
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = max(remaining.rfind(". ", 0, limit), remaining.rfind("; ", 0, limit), remaining.rfind(" ", 0, limit))
        if cut < limit // 2:
            cut = limit
        pieces.append(remaining[:cut + 1].strip())
        remaining = remaining[cut + 1:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _document_metadata(soup: BeautifulSoup) -> tuple[str, int, str]:
    h1 = _clean_text(soup.find("h1"))
    company = re.sub(r"\s*\([^)]*(?::| ticker)?[^)]*\)\s*$", "", h1, flags=re.IGNORECASE).strip()
    company = re.sub(r"\s*\([A-Z]{1,8}\)\s*$", "", company).strip()
    text = _clean_text(soup.find("body") or soup)
    if re.search(r"Annual Report\s+20\d{2}", text, re.IGNORECASE):
        doc_type = "annual_report"
        match = re.search(r"Annual Report\s+(20\d{2})", text, re.IGNORECASE)
    elif re.search(r"Earnings (?:Conference )?Call", text, re.IGNORECASE):
        doc_type = "earnings_call"
        match = re.search(r"Full Year\s+(20\d{2})|FY(20\d{2})", text, re.IGNORECASE)
    else:
        doc_type = "research_report"
        match = re.search(r"FY(20\d{2})\s+results review", text, re.IGNORECASE)
    if not match:
        raise ValueError("Could not determine document fiscal year")
    fiscal_year = int(next(group for group in match.groups() if group))
    return company, fiscal_year, doc_type


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if table is None:
        return ""
    caption_node = table.find("caption")
    caption = _clean_text(caption_node) if caption_node else ""
    header_rows = table.select("thead tr")
    headers = [_clean_text(cell) for cell in header_rows[-1].find_all(["th", "td"], recursive=False)] if header_rows else []
    lines = [caption] if caption else []
    body_rows = table.select("tbody tr") or [row for row in table.find_all("tr") if row not in header_rows]
    for row in body_rows:
        cells = [_clean_text(cell) for cell in row.find_all(["th", "td"], recursive=False)]
        if not cells or not any(cells):
            continue
        row_label = cells[0]
        values = []
        for column, value in enumerate(cells[1:], 1):
            if not value:
                continue
            header = headers[column] if column < len(headers) and headers[column] else f"Column {column + 1}"
            values.append(f"{header}: {value}")
        if not values:
            lines.append(row_label)
            continue
        line = f"{row_label} | " + " | ".join(values)
        if len(line) <= 400:
            lines.append(line)
        else:
            lines.extend(f"{row_label} | {value}" for value in values)
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = _document_metadata(soup)
    display_name = re.sub(r"\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC)$", "", company, flags=re.IGNORECASE)
    _CORPUS_COMPANIES[_company_key_local(display_name)] = display_name
    base_metadata = {"source": file_name, "company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}
    chunks: list[Document] = []

    def add(text: str, section: str, kind: str = "narrative") -> None:
        text = text.strip()
        if not text:
            return
        # Keep the complete original text; only very unusual long table rows need hard splitting.
        for start in range(0, len(text), 1450):
            part = text[start:start + 1450]
            chunks.append(Document(page_content=part, metadata={**base_metadata, "section": section, "kind": kind}))

    body = soup.find("body") or soup
    section = "Document information"
    narrative: list[str] = []

    def flush() -> None:
        nonlocal narrative
        if narrative:
            text = "\n".join(narrative)
            for part in _split_narrative(text):
                add(f"Section: {section}\n{part}", section)
            narrative = []

    for element in body.find_all(["h1", "h2", "h3", "p", "li", "table"], recursive=True):
        if element.find_parent(["p", "li", "table"]):
            continue
        if element.name in ("h1", "h2", "h3"):
            flush()
            section = _clean_text(element)
            if element.name == "h1":
                narrative.append(section)
        elif element.name == "table":
            flush()
            table_text = table_to_text(str(element))
            table_lines = table_text.splitlines()
            caption = table_lines[0] if table_lines else section
            for line in table_lines[1:]:
                add(f"Section: {section}\n{caption}\n{line}", section, "table")
        else:
            text = _clean_text(element)
            if text:
                narrative.append(text)
    flush()
    return chunks


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    lowered = question.casefold()
    if re.search(r"earnings (?:conference )?call|conference call|\bq[1-4]\b|fourth-quarter", lowered):
        doc_type = "earnings_call"
    elif re.search(r"research report|results review|target price|investment rating|\brating\b|\bforecast|\bestimate", lowered):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", question)
    fiscal_year = int(year_match.group(1)) if year_match else None

    companies: list[str] = []
    for key, name in sorted(_CORPUS_COMPANIES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?:\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC|LLC))?(?![A-Za-z0-9])",
                     question, re.IGNORECASE):
            companies.append(name)
    matched_keys = {_company_key_local(name) for name in companies}
    for candidate in _unknown_company_candidates(question):
        key = _company_key_local(candidate)
        if key not in matched_keys and _is_company_candidate(candidate):
            companies.append(candidate)
            matched_keys.add(key)
    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    tokenized = [_tokens(chunk.page_content) for chunk in chunks]
    by_company: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        by_company.setdefault(_company_key_local(str(chunk.metadata.get("company", ""))), []).append(i)
    years = sorted({int(chunk.metadata["fiscal_year"]) for chunk in chunks if chunk.metadata.get("fiscal_year") is not None})
    return {"chunks": chunks, "bm25": BM25Okapi(tokenized), "by_company": by_company,
            "min_year": min(years), "max_year": max(years)}


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    parsed = parse_query(question)
    known = {_company_key_local(name): name for name in parsed["companies"] if _company_key_local(name) in index["by_company"]}
    if not known or len(known) != len(parsed["companies"]):
        return []
    doc_type = parsed["doc_type"]
    year = parsed["fiscal_year"]
    if doc_type == "annual_report" and year is not None and not (index["min_year"] - 1 <= year <= index["max_year"]):
        return []

    candidate_ids = [i for key in known for i in index["by_company"][key]
                     if index["chunks"][i].metadata.get("doc_type") == doc_type]
    if not candidate_ids:
        return []

    # If distinctive modifiers (not ordinary financial terminology) are absent from the relevant
    # documents, abstain rather than returning a merely similar financial statement.
    concepts = _query_concepts(question)
    reduced = _strip_known_companies(question)
    expansion = [word for concept in concepts for word in concept]
    concept_words = {word for triggers, output in _CONCEPTS for word in (*triggers, *output)}
    ignored = _QUERY_STOPWORDS | concept_words | {"large", "month", "management", "guide", "guided", "announce", "announced", "city", "hold", "forecast", "estimate"}
    distinctive = [word for word in _tokens(reduced)
                   if word not in ignored and not word.isdigit() and len(word) >= 4
                   and not re.fullmatch(r"fy\d{4}", word)]
    corpus_text = " ".join(index["chunks"][i].page_content.casefold() for i in candidate_ids)
    corpus_words = set(_tokens(corpus_text))
    if any(word not in corpus_words for word in distinctive):
        return []

    # Company filters already establish identity; removing the names keeps them from overwhelming
    # the metric terms when two companies are ranked independently for a comparison. Alias words
    # are replaced by their expanded concepts so "based" cannot rank "based on" above headquarters.
    aliases = {"based", "located", "sales", "turnover", "staff", "workforce", "headcount", "cfo", "ceo"}
    query_words = [word for word in _tokens(reduced) if word not in aliases]
    scores = index["bm25"].get_scores(query_words + expansion)
    question_terms = {word for word in query_words if word not in _QUERY_STOPWORDS and not word.isdigit()}

    def adjusted_score(i: int) -> float:
        chunk = index["chunks"][i]
        score = float(scores[i])
        words = set(_tokens(chunk.page_content))
        score += 0.7 * len(question_terms & words)
        if chunk.metadata.get("kind") == "table" and concepts:
            score += 1.0
        if doc_type == "annual_report" and year is not None:
            document_year = int(chunk.metadata["fiscal_year"])
            if document_year == year:
                score += 5.0
            elif document_year == year + 1:
                score += 1.0
            else:
                score -= 20.0
        return score

    selected: list[Document] = []
    per_company = max(1, 5 // len(known))
    for key in known:
        ids = [i for i in candidate_ids if _company_key_local(str(index["chunks"][i].metadata["company"])) == key]
        ranked = sorted(ids, key=lambda i: (-adjusted_score(i), i))
        # Prefer the exact-year report whenever the requested year has one; the following year's
        # report is the necessary source for the earliest comparative column.
        if doc_type == "annual_report" and year is not None:
            exact = [i for i in ranked if int(index["chunks"][i].metadata["fiscal_year"]) == year]
            if exact:
                ranked = exact
        selected.extend(index["chunks"][i] for i in ranked[:per_company])
    if len(selected) < 5:
        chosen = {id(chunk) for chunk in selected}
        remainder = sorted((i for i in candidate_ids if id(index["chunks"][i]) not in chosen),
                           key=lambda i: (-adjusted_score(i), i))
        selected.extend(index["chunks"][i] for i in remainder[:5 - len(selected)])
    return selected[:5]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()

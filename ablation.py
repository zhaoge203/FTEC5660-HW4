"""Reproducible B0--V8 retrieval ablation for FTEC5660 HW4.

Run from this directory (DEEPSEEK_API_KEY is supplied by the environment):
    .venv/Scripts/python ablation.py --data-folder public_test --output ablation_results.csv

Only the fixed hw4.generate function makes model calls. Chunking, indexing,
retrieval, and scoring are deterministic and use stable corpus-order tie breaks.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


Retriever = Callable[[Any, str], list[Document]]
WORKERS = 8


def recursive_chunks(file_name: str, html: str, convert_tables: bool) -> list[Document]:
    """B0/B1: BeautifulSoup get_text followed only by RecursiveCharacterTextSplitter."""
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = hw4._document_metadata(soup)
    if convert_tables:
        for table in list(soup.find_all("table")):
            table_text = hw4.table_to_text(str(table))
            table.replace_with(NavigableString(table_text))

    # Deliberately no separator argument and no contextual headers in B0/B1.
    text = soup.get_text()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=hw4.MAX_CHUNK_CHARS,
        chunk_overlap=0,
    )
    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
    }
    return splitter.create_documents([text], metadatas=[metadata])


Tokenizer = Callable[[str], list[str]]


def build_naive_index(chunks: list[Document]) -> dict[str, Any]:
    """Lower-whitespace BM25, with metadata groups unused until V3 filtering."""
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    by_company: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        key = hw4._company_key_local(str(chunk.metadata.get("company", "")))
        by_company.setdefault(key, []).append(i)
    return {"chunks": chunks, "bm25": BM25Okapi(tokenized), "by_company": by_company}


def naive_retrieve(index: dict[str, Any], question: str) -> list[Document]:
    """Full-corpus BM25 top five, without parsing, filtering, or abstention."""
    scores = index["bm25"].get_scores(question.lower().split())
    ranked = sorted(range(len(index["chunks"])), key=lambda i: (-float(scores[i]), i))
    return [index["chunks"][i] for i in ranked[: hw4.MAX_RETRIEVED]]


def _metadata_candidate_groups(index: dict[str, Any], parsed: dict[str, Any]) -> tuple[dict[str, str], list[list[int]]]:
    """V3 filtering only; relax missing companies/types/years instead of rejecting."""
    chunks = index["chunks"]
    all_ids = list(range(len(chunks)))
    known: dict[str, str] = {}
    for name in parsed["companies"]:
        key = hw4._company_key_local(name)
        if key in index["by_company"]:
            known[key] = name

    if known:
        company_ids = [i for key in known for i in index["by_company"][key]]
    else:
        # Unknown company: retain a doc_type-relevant full-corpus fallback.
        company_ids = all_ids

    doc_type = parsed["doc_type"]
    candidate_ids = [i for i in company_ids if chunks[i].metadata.get("doc_type") == doc_type]
    if not candidate_ids:
        # A known company without that doc_type still gets its best related candidates.
        candidate_ids = company_ids
    if not candidate_ids:
        candidate_ids = all_ids

    year = parsed["fiscal_year"]
    groups: list[list[int]] = []
    keys = list(known) if known else [None]
    for key in keys:
        ids = [
            i for i in candidate_ids
            if key is None or hw4._company_key_local(str(chunks[i].metadata["company"])) == key
        ]
        # Exact-year filtering/preference belongs to V3. If the requested year is
        # absent, retain other years rather than turning this into abstention.
        if doc_type == "annual_report" and year is not None:
            exact = [i for i in ids if int(chunks[i].metadata["fiscal_year"]) == year]
            if exact:
                ids = exact
        groups.append(ids)
    return known, groups


def metadata_retrieve(
    index: dict[str, Any],
    question: str,
    query_tokenizer: Tokenizer,
    synonym_expansion: bool = False,
    ranking_boosts: bool = False,
    comparison_quotas: bool = False,
) -> list[Document]:
    """V3--V7 retrieval, with each post-filtering component explicitly gated."""
    parsed = hw4.parse_query(question)
    known, groups = _metadata_candidate_groups(index, parsed)
    candidate_ids = list(dict.fromkeys(i for group in groups for i in group))

    # Company-name stripping is part of the V3 metadata-aware query handling.
    reduced = hw4._strip_known_companies(question)
    query_words = query_tokenizer(reduced)
    concepts = hw4._query_concepts(question) if synonym_expansion else []
    expansion = [word for concept in concepts for word in concept]
    if synonym_expansion:
        # Alias replacement is part of V5's concept expansion, so V8 adds only abstention.
        aliases = {"based", "located", "sales", "turnover", "staff", "workforce", "headcount", "cfo", "ceo"}
        query_words = [word for word in query_words if word not in aliases]
    scores = index["bm25"].get_scores(query_words + expansion)
    doc_type = parsed["doc_type"]
    year = parsed["fiscal_year"]

    def ranking_score(i: int) -> float:
        score = float(scores[i])
        if not ranking_boosts:
            return score

        # V6 only: the final hw4 overlap/table/exact-year adjusted-score boosts.
        chunk = index["chunks"][i]
        question_terms = {
            word for word in query_words
            if word not in hw4._QUERY_STOPWORDS and not word.isdigit()
        }
        words = set(hw4._tokens(chunk.page_content))
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

    def ranked(ids: list[int]) -> list[int]:
        return sorted(ids, key=lambda i: (-ranking_score(i), i))

    # Through V6, all filtered candidates are ranked globally without quotas.
    if not comparison_quotas or len(known) < 2:
        return [index["chunks"][i] for i in ranked(candidate_ids)[: hw4.MAX_RETRIEVED]]

    # V7 only: independent comparison quotas, then the global best fifth chunk.
    selected: list[Document] = []
    per_company = max(1, hw4.MAX_RETRIEVED // len(known))  # two companies: 2 + 2
    for group in groups:
        selected.extend(index["chunks"][i] for i in ranked(group)[:per_company])
    if len(selected) < hw4.MAX_RETRIEVED:
        chosen = {id(chunk) for chunk in selected}
        # Match hw4.retrieve exactly: the fifth chunk may come from either annual-report
        # year, even though the guaranteed per-company slots prefer the exact year.
        remainder_pool = [
            i for key in known for i in index["by_company"][key]
            if index["chunks"][i].metadata.get("doc_type") == doc_type
        ]
        remainder = [i for i in remainder_pool if id(index["chunks"][i]) not in chosen]
        selected.extend(
            index["chunks"][i]
            for i in ranked(remainder)[: hw4.MAX_RETRIEVED - len(selected)]
        )
    return selected[: hw4.MAX_RETRIEVED]


def canonical_retrieval(
    result: Any,
    by_text: dict[str, list[Document]],
) -> tuple[list[Document], bool]:
    """Apply the official runner's validity and source-canonicalization rules."""
    ok = isinstance(result, (list, tuple)) and len(result) <= hw4.MAX_RETRIEVED
    kept: list[Document] = []
    iterable = list(result or []) if isinstance(result, (list, tuple)) else []
    for chunk in iterable:
        matches = by_text.get(getattr(chunk, "page_content", None), [])
        if not matches:
            ok = False
            continue
        source = (getattr(chunk, "metadata", None) or {}).get("source")
        kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
    return kept[: hw4.MAX_RETRIEVED], ok


def evaluate_variant(
    name: str,
    change: str,
    chunks: list[Document],
    index: Any,
    retriever: Retriever,
    questions: dict[str, str],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve, generate with the official fixed generator, and compute official metrics."""
    print(f"\n{name}: {change}")
    print(f"  {len(chunks)} chunks; retrieving {len(questions)} questions")
    by_text: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_text.setdefault(chunk.page_content, []).append(chunk)

    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        result = retriever(index, question)
        retrieved[qid], valid[qid] = canonical_retrieval(result, by_text)

    def answer(qid: str) -> str:
        try:
            return hw4.generate(questions[qid], retrieved[qid])
        except Exception as exc:
            print(f"  Generator failed on {qid}: {exc!r}")
            return ""

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))

    graded = [qid for qid in questions if qid in ground_truth]
    answerable = [qid for qid in graded if ground_truth[qid]["evidence"]]
    unanswerable = [qid for qid in graded if not ground_truth[qid]["evidence"]]
    if not graded:
        raise ValueError("No questions have public ground truth")

    valid_nonempty = sum(
        valid[qid] and bool(retrieved[qid]) for qid in answerable
    ) / max(len(answerable), 1)
    recall = sum(
        hw4.retrieval_hit(retrieved[qid], ground_truth[qid]) for qid in answerable
    ) / max(len(answerable), 1)
    kept = sum(bool(retrieved[qid]) for qid in answerable) / max(len(answerable), 1)
    rejected = sum(not retrieved[qid] for qid in unanswerable) / max(len(unanswerable), 1)
    answer_accuracy = sum(
        hw4.answer_correct(answers[qid], ground_truth[qid]) for qid in graded
    ) / len(graded)

    row: dict[str, Any] = {
        "variant": name,
        "change": change,
        "chunks": len(chunks),
        "valid_nonempty": valid_nonempty,
        "recall": recall,
        "kept": kept,
        "rejected": rejected,
        "answer_accuracy": answer_accuracy,
    }
    print(
        f"  valid_nonempty={valid_nonempty:.1%} recall={recall:.1%} "
        f"kept={kept:.1%} rejected={rejected:.1%} answer_accuracy={answer_accuracy:.1%}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="B0--V8 ablation using the official HW4 scorer")
    parser.add_argument(
        "--data-folder",
        default="public_test",
        help="folder containing corpus/, questions.json and ground_truth.json",
    )
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")
    if not corpus or not isinstance(questions, dict) or not questions:
        raise ValueError("The data folder must contain a non-empty corpus and questions.json")
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("This ablation requires public_test/ground_truth.json")

    b0_chunks: list[Document] = []
    v1_chunks: list[Document] = []
    structured_chunks: list[Document] = []
    for file_name in sorted(corpus):
        html = corpus[file_name]
        b0_chunks.extend(recursive_chunks(file_name, html, convert_tables=False))
        v1_chunks.extend(recursive_chunks(file_name, html, convert_tables=True))
        structured_chunks.extend(hw4.chunk_document(file_name, html))

    for chunks in (b0_chunks, v1_chunks, structured_chunks):
        if not chunks:
            raise ValueError("Chunking produced no chunks")
        oversized = max(len(chunk.page_content) for chunk in chunks)
        if oversized > hw4.MAX_CHUNK_CHARS:
            raise ValueError(f"A chunk has {oversized} characters (limit {hw4.MAX_CHUNK_CHARS})")

    b0_index = build_naive_index(b0_chunks)
    v1_index = build_naive_index(v1_chunks)
    # V2 and V3 share the exact same lower-whitespace BM25 index. V4 is the
    # first variant to use hw4._tokens for both the corpus and the query.
    whitespace_structured_index = build_naive_index(structured_chunks)
    hw4_token_index = hw4.build_index(structured_chunks)

    variants = [
        (
            "B0",
            "BeautifulSoup get_text + Recursive splitter + naive full-corpus BM25",
            b0_chunks,
            b0_index,
            naive_retrieve,
        ),
        (
            "V1",
            "+ hw4.table_to_text table replacement only",
            v1_chunks,
            v1_index,
            naive_retrieve,
        ),
        (
            "V2",
            "+ hw4 section/table-row chunks and contextual headers only",
            structured_chunks,
            whitespace_structured_index,
            naive_retrieve,
        ),
        (
            "V3",
            "+ query parsing + company/doc_type/exact-year metadata filtering only",
            structured_chunks,
            whitespace_structured_index,
            lambda index, question: metadata_retrieve(
                index,
                question,
                query_tokenizer=lambda text: text.lower().split(),
                synonym_expansion=False,
                ranking_boosts=False,
                comparison_quotas=False,
            ),
        ),
        (
            "V4",
            "+ hw4._tokens corpus/query tokenizer only",
            structured_chunks,
            hw4_token_index,
            lambda index, question: metadata_retrieve(
                index,
                question,
                query_tokenizer=hw4._tokens,
                synonym_expansion=False,
                ranking_boosts=False,
                comparison_quotas=False,
            ),
        ),
        (
            "V5",
            "+ concept synonym expansion only",
            structured_chunks,
            hw4_token_index,
            lambda index, question: metadata_retrieve(
                index,
                question,
                query_tokenizer=hw4._tokens,
                synonym_expansion=True,
                ranking_boosts=False,
                comparison_quotas=False,
            ),
        ),
        (
            "V6",
            "+ overlap/table/exact-year adjusted_score ranking boosts only",
            structured_chunks,
            hw4_token_index,
            lambda index, question: metadata_retrieve(
                index,
                question,
                query_tokenizer=hw4._tokens,
                synonym_expansion=True,
                ranking_boosts=True,
                comparison_quotas=False,
            ),
        ),
        (
            "V7",
            "+ per-company comparison quotas (2+2+global best fifth) only",
            structured_chunks,
            hw4_token_index,
            lambda index, question: metadata_retrieve(
                index,
                question,
                query_tokenizer=hw4._tokens,
                synonym_expansion=True,
                ranking_boosts=True,
                comparison_quotas=True,
            ),
        ),
        (
            "V8",
            "direct hw4.retrieve; + final abstention only",
            structured_chunks,
            hw4_token_index,
            hw4.retrieve,
        ),
    ]

    rows = [
        evaluate_variant(name, change, chunks, index, retriever, questions, ground_truth)
        for name, change, chunks, index, retriever in variants
    ]
    fieldnames = [
        "variant", "change", "chunks", "valid_nonempty", "recall",
        "kept", "rejected", "answer_accuracy",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })

    print("\nAblation summary")
    print(f"{'variant':<8} {'chunks':>6} {'valid_nonempty':>15} {'recall':>8} {'kept':>8} {'rejected':>9} {'answer_accuracy':>16}")
    for row in rows:
        print(
            f"{row['variant']:<8} {row['chunks']:>6} {row['valid_nonempty']:>15.1%} "
            f"{row['recall']:>8.1%} {row['kept']:>8.1%} {row['rejected']:>9.1%} "
            f"{row['answer_accuracy']:>16.1%}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

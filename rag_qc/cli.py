import argparse, os
from tqdm import tqdm

from .config import settings
from .utils import ensure_dir, sha1
from .ingest import load_raw_documents
from .chunking import make_chunks
from .embeddings import get_provider
from .index import build_index, save_index, load_index, exists
from .retrieve import search
from .judge import judge
from .answer import extractive_answer

def cmd_build():
    ensure_dir(settings.raw_dir)
    ensure_dir(settings.index_dir)

    pages = load_raw_documents(settings.raw_dir)
    if not pages:
        print(f"No hay PDFs en {settings.raw_dir}. Poné PDFs ahí y reintentá.")
        return

    chunks = make_chunks(pages, id_fn=sha1)
    meta = [{"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "text": c.text} for c in chunks]

    provider = get_provider()
    vectors = provider.embed([c["text"] for c in meta])

    index = build_index(vectors)
    save_index(index, meta, settings.index_dir)

    print(f"Index construido: {len(meta)} chunks. Guardado en {settings.index_dir}")

def cmd_ask(question: str, k: int):
    if not exists(settings.index_dir):
        print("No existe índice. Ejecutá: python -m rag_qc.cli build")
        return

    index, meta = load_index(settings.index_dir)
    provider = get_provider()
    qvec = provider.embed([question])[0]

    retrieved = search(index, meta, qvec, k=k)
    j = judge(retrieved)
    out = extractive_answer(question, retrieved, j)

    print("\n=== RESULT ===")
    print(f"STATUS: {out.status}")
    print(f"CONFIDENCE: {out.confidence:.2f}")
    if out.reasons:
        print("REASONS:")
        for r in out.reasons:
            print(f"- {r}")
    print("\nANSWER:\n")
    print(out.answer)
    print("\nCITATIONS:")
    for c in out.citations:
        print(f"- {c['source']} p.{c['page']} (chunk={c['chunk_id'][:8]} score={c['score']:.3f})")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("--q", required=True)
    a.add_argument("--k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "build":
        cmd_build()
    elif args.cmd == "ask":
        cmd_ask(args.q, args.k)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

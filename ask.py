"""Ask questions from the terminal.

Usage:  python ask.py              (interactive)
        python ask.py "question"   (one-shot)
"""
import sys

from rag.pipeline import answer
from rag.vectorstore import load_index


def show(result):
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['source']} p.{s['page']}  (distance {s['score']})")
    print()


def main():
    index = load_index()
    if len(sys.argv) > 1:
        show(answer(index, " ".join(sys.argv[1:])))
        return
    print("Ask a question (empty line to quit).")
    while True:
        q = input("> ").strip()
        if not q:
            break
        show(answer(index, q))


if __name__ == "__main__":
    main()

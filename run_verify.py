#!/usr/bin/env python3
"""CLI wrapper for the Limitless footage checker (GitHub Actions entrypoint).

Usage: python run_verify.py PAYLOAD_JSON OUT_JSON
  PAYLOAD_JSON: {"line": str, "candidates": [{"url", "title"}, ...]}
  OUT_JSON:     written with app.verify()'s JSON result:
                {"line": ..., "results": [{url,title,score,neg,band,error?}]}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    with open(argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)
    line = payload.get("line", "")
    candidates = payload.get("candidates", [])
    result = app.verify(line, json.dumps(candidates))
    with open(argv[2], "w", encoding="utf-8") as f:
        f.write(result)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

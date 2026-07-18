"""Local web interface for the fashion retrieval project.

Run from the project root with: ``python webapp/server.py``.
The server intentionally uses only Python's standard library. Model imports are
deferred until a search is requested, so the interface can explain setup issues
without failing at startup.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "test"

sys.path.insert(0, str(PROJECT_ROOT))


def serialize_parsed_query(query: str) -> dict:
    from retriever.query_parser import QueryParser

    parsed = QueryParser().parse(query)
    return {
        "pairs": [{"garment": pair.garment, "color": pair.color} for pair in parsed.pairs],
        "environmentHits": parsed.environment_hits,
        "styleHits": parsed.style_hits,
    }


def explain_result(result: dict, parsed_query: dict) -> dict:
    garments = result["garments"]
    matched_pairs = []
    partial_pairs = []

    for pair in parsed_query["pairs"]:
        exact = next(
            (item for item in garments if item["type"] == pair["garment"] and item["color"] == pair["color"]),
            None,
        )
        related = next((item for item in garments if item["type"] == pair["garment"]), None)
        label = " ".join(value for value in (pair["color"], pair["garment"]) if value)
        if exact:
            matched_pairs.append(label)
        elif related:
            partial_pairs.append(label)

    context_matches = [
        value
        for value in [*parsed_query["environmentHits"], *parsed_query["styleHits"]]
        if value in {result["environment"], result["style"]}
    ]
    return {
        "matchedPairs": matched_pairs,
        "partialPairs": partial_pairs,
        "contextMatches": context_matches,
    }


class FashionRequestHandler(SimpleHTTPRequestHandler):
    index_dir = DEFAULT_INDEX
    images_dir = DEFAULT_IMAGES

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        if request.path == "/api/health":
            self.handle_health()
            return
        if request.path == "/api/parse":
            self.handle_parse(request.query)
            return
        if request.path == "/api/search":
            self.handle_search(request.query)
            return
        if request.path.startswith("/images/"):
            self.handle_image(request.path)
            return
        self.serve_static(request.path)

    def handle_health(self) -> None:
        missing = []
        if not (self.index_dir / "index.faiss").is_file():
            missing.append("data/index/index.faiss")
        if not self.images_dir.is_dir():
            missing.append("data/test")
        self.send_json(
            HTTPStatus.OK,
            {
                "ready": not missing,
                "missing": missing,
                "message": "Search is ready." if not missing else "Complete the local model/index setup to enable searches.",
            },
        )

    def handle_parse(self, raw_query: str) -> None:
        query = parse_qs(raw_query).get("q", [""])[0].strip()
        if not query:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Enter a search phrase."})
            return
        try:
            self.send_json(HTTPStatus.OK, {"query": query, "parsedQuery": serialize_parsed_query(query)})
        except Exception as exc:  # setup errors should remain actionable in the UI
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def handle_search(self, raw_query: str) -> None:
        params = parse_qs(raw_query)
        query = params.get("q", [""])[0].strip()
        if not query:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Enter a search phrase."})
            return

        try:
            parsed_query = serialize_parsed_query(query)
            if not (self.index_dir / "index.faiss").is_file():
                raise RuntimeError("The FAISS index is missing. Run the indexer before searching.")

            from retriever.search import Retriever

            top_k = min(max(int(params.get("top_k", ["12"])[0]), 1), 24)
            results = Retriever(self.index_dir).search(query, top_k=top_k)
            response_results = []
            for result in results:
                image_path = self.images_dir / result["image_id"]
                response_results.append(
                    {
                        **result,
                        "imageUrl": f"/images/{quote(result['image_id'])}" if image_path.is_file() else None,
                        "explanation": explain_result(result, parsed_query),
                    }
                )
            self.send_json(HTTPStatus.OK, {"query": query, "parsedQuery": parsed_query, "results": response_results})
        except (ImportError, ModuleNotFoundError) as exc:
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": f"Model dependencies are unavailable: {exc}. Install requirements.txt, then build the index."},
            )
        except Exception as exc:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})

    def handle_image(self, request_path: str) -> None:
        image_name = Path(unquote(request_path.removeprefix("/images/"))).name
        image_path = self.images_dir / image_name
        if image_name != unquote(request_path.removeprefix("/images/")) or not image_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        data = image_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_static(self, request_path: str) -> None:
        relative_path = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_ROOT / relative_path).resolve()
        if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Glance fashion retrieval web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--index", default=DEFAULT_INDEX, type=Path)
    parser.add_argument("--images", default=DEFAULT_IMAGES, type=Path)
    args = parser.parse_args()

    FashionRequestHandler.index_dir = args.index.resolve()
    FashionRequestHandler.images_dir = args.images.resolve()
    httpd = ThreadingHTTPServer((args.host, args.port), FashionRequestHandler)
    print(f"Glance is available at http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

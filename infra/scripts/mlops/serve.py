#!/usr/bin/env python3
"""Serve a local reference model over a small localhost-only HTTP API."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .model import load_model, predict
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlops.model import load_model, predict


MAX_BODY_BYTES = 1024 * 1024


def _metadata(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model.get("model_name"),
        "model_version": model.get("model_version"),
        "format": model.get("format"),
        "labels": model.get("labels", []),
    }


def create_server(
    model_path: Path, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    model = load_model(model_path.resolve())
    counters = {"requests": 0, "predictions": 0, "errors": 0}

    def _prometheus_label(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "unknown"))

    class Handler(BaseHTTPRequestHandler):
        server_version = "SoloEmpireMLops/1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_metrics(self) -> None:
            labels = (
                f'model_name="{_prometheus_label(model.get("model_name"))}",'
                f'model_version="{_prometheus_label(model.get("model_version"))}"'
            )
            body = (
                "# HELP solo_empire_mlops_requests_total HTTP requests received.\n"
                "# TYPE solo_empire_mlops_requests_total counter\n"
                f"solo_empire_mlops_requests_total{{{labels}}} {counters['requests']}\n"
                "# HELP solo_empire_mlops_predictions_total Successful predictions.\n"
                "# TYPE solo_empire_mlops_predictions_total counter\n"
                f"solo_empire_mlops_predictions_total{{{labels}}} {counters['predictions']}\n"
                "# HELP solo_empire_mlops_errors_total Invalid or failed requests.\n"
                "# TYPE solo_empire_mlops_errors_total counter\n"
                f"solo_empire_mlops_errors_total{{{labels}}} {counters['errors']}\n"
                "# HELP solo_empire_mlops_model_info Loaded model metadata.\n"
                "# TYPE solo_empire_mlops_model_info gauge\n"
                f"solo_empire_mlops_model_info{{{labels}}} 1\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/readyz":
                self._send_json(200, {"status": "ready", **_metadata(model)})
            elif self.path == "/model":
                self._send_json(200, _metadata(model))
            elif self.path == "/metrics":
                self._send_metrics()
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            counters["requests"] += 1
            if self.path != "/predict":
                counters["errors"] += 1
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                counters["errors"] += 1
                self._send_json(413, {"error": "request body is too large or missing"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                text = (
                    str(payload.get("text") or "") if isinstance(payload, dict) else ""
                )
                if not text.strip():
                    raise ValueError("text is required")
                prediction, scores = predict(model, text)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                counters["errors"] += 1
                self._send_json(400, {"error": str(exc)})
                return
            counters["predictions"] += 1
            self._send_json(
                200,
                {
                    **_metadata(model),
                    "prediction": prediction,
                    "scores": {
                        label: round(score, 6)
                        for label, score in sorted(scores.items())
                    },
                },
            )

        def log_message(self, _format: str, *_args: Any) -> None:
            # Do not emit request paths or payload-adjacent data by default.
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.model = model  # type: ignore[attr-defined]
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local MLOps reference model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    server = create_server(args.model, args.host, args.port)
    print(
        json.dumps(
            {"status": "listening", "host": args.host, "port": server.server_address[1]}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

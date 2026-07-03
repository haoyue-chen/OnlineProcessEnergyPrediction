"""Minimal HTTP inference server for the MoE energy predictor.

Stdlib only (http.server) so the container needs nothing beyond numpy + sklearn.

Endpoints:
  GET  /health           -> {"status": "ok"}
  GET  /info             -> model metadata (features, labels, expert class)
  POST /predict          -> {"features": {...}} -> {"energy_wh": float, "expert": label}
  POST /predict_batch    -> {"samples": [{...}, ...]} -> {"predictions": [...]}

Config via env:
  MODEL_PATH  (default /app/models/moe_linear.pkl)
  PORT        (default 8800)
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .predictor import EnergyPredictor

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/moe_linear.pkl")
PORT = int(os.environ.get("PORT", "8800"))

predictor = EnergyPredictor(MODEL_PATH)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def log_message(self, *args):  # quieter logs
        return

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/info":
            self._send(200, predictor.info())
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/predict":
                feats = data.get("features", data)
                energy = predictor.predict_sample(feats)
                self._send(200, {"energy_wh": energy, "expert": predictor.route(feats)})
            elif self.path == "/predict_batch":
                samples = data.get("samples", [])
                self._send(200, {"predictions": predictor.predict_batch(samples)})
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:  # surface errors as JSON, don't crash the server
            self._send(400, {"error": str(exc)})


def main():
    print(f"Loading model: {MODEL_PATH}")
    print(f"Model info: {predictor.info()['expert_class']} expert, "
          f"{predictor.info()['n_features']} features, labels={predictor.labels}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on 0.0.0.0:{PORT} (GET /health /info ; POST /predict /predict_batch)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

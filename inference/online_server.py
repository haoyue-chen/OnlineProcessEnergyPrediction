"""HTTP server for the live online-learning MoE service.

Stdlib only. Endpoints:
  GET  /health              -> {"status": "ok"}
  GET  /info                -> model metadata + num_updates + model_version
  POST /predict             -> {"features": {...}}
                               -> {prediction_id, energy_wh, expert, model_version}
  POST /update              -> {"prediction_id"|"features", "true_energy_wh", ["expert"]}
                               -> {updated_expert, num_updates, model_version}
  POST /predict_then_update -> {"features": {...}, "true_energy_wh": float}

Config via env:
  ONLINE_BASE_PATH   (default /app/models/online_base.pkl)  immutable warm-started base
  ONLINE_STATE_PATH  (default /app/models/online_state.pkl) evolving state (persist via volume)
  PORT               (default 8800)
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .online_predictor import OnlinePredictor

BASE_PATH = os.environ.get("ONLINE_BASE_PATH", "/app/models/online_base.pkl")
STATE_PATH = os.environ.get("ONLINE_STATE_PATH", "/app/models/online_state.pkl")
PORT = int(os.environ.get("PORT", "8800"))

predictor = OnlinePredictor(BASE_PATH, STATE_PATH)


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
        return json.loads(self.rfile.read(length).decode()) if length else {}

    def log_message(self, *args):
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
                self._send(200, predictor.predict(feats))
            elif self.path == "/update":
                self._send(200, predictor.update(
                    true_energy_wh=data["true_energy_wh"],
                    prediction_id=data.get("prediction_id"),
                    features=data.get("features"),
                    expert=data.get("expert"),
                ))
            elif self.path == "/predict_then_update":
                self._send(200, predictor.predict_then_update(
                    data.get("features", {}), data["true_energy_wh"]))
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except KeyError as exc:
            self._send(400, {"error": f"missing field: {exc}"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})


def main():
    info = predictor.info()
    print(f"Online learning service — expert={info['online_expert']}, "
          f"loaded_from={info['loaded_from']}, "
          f"num_updates={info['num_updates']}, version={info['model_version']}")
    print(f"  base={BASE_PATH}")
    print(f"  state={STATE_PATH} (persist this via a mounted volume)")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on 0.0.0.0:{PORT} "
          f"(GET /health /info ; POST /predict /update /predict_then_update)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

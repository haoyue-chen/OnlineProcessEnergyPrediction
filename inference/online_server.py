"""HTTP server for the live online-learning MoE service.

Stdlib only. Endpoints:
  GET  /health              -> {"status": "ok"}
  GET  /info                -> model metadata + model_version
  POST /predict             -> {"features": {...}}
                               -> {prediction_id, energy_wh, expert, model_version, ...}
  POST /update              -> {"prediction_id"|"features", "true_energy_wh", ["expert"]}
                               -> {updated_expert, num_updates, model_version}
  POST /predict_then_update -> {"features": {...}, "true_energy_wh": float}

Serving safety: /predict serves ONLY approved models selected via
``models/latest_approved.json`` (ApprovedPredictor). Pending or rejected
candidates under ``models/pending/`` are never loaded. If no approved model is
registered, /predict falls back safely to the legacy base MoE artifact
(``models/moe_linear.pkl``). The /update and /predict_then_update endpoints keep
operating on the legacy online-learning state (OnlinePredictor); they do not
mutate the approved model.

Config via env:
  ONLINE_BASE_PATH   (default /app/models/online_base.pkl)  immutable warm-started base (online state)
  ONLINE_STATE_PATH  (default /app/models/online_state.pkl) evolving online state (persist via volume)
  PORT               (default 8800)
  ONLINE_PRED_LOG    optional .jsonl path for prediction logging
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .approved_predictor import ApprovedPredictor
from .online_predictor import OnlinePredictor

BASE_PATH = os.environ.get("ONLINE_BASE_PATH", "/app/models/online_base.pkl")
STATE_PATH = os.environ.get("ONLINE_STATE_PATH", "/app/models/online_state.pkl")
PORT = int(os.environ.get("PORT", "8800"))
# Optional prediction logging for the auto-expert pipeline. Unset -> no logging,
# so default behaviour is unchanged. Set to a .jsonl path to enable.
PRED_LOG = os.environ.get("ONLINE_PRED_LOG", "")

# /predict source of truth: only approved models (with safe base fallback).
approved = ApprovedPredictor()
# Online-learning state for /update /predict_then_update (legacy path; does not
# touch the approved model). Constructed lazily-safe: if the online base/state
# artifacts are absent in a non-online deployment, /update simply errors per-call.
online: OnlinePredictor | None = None
try:
    online = OnlinePredictor(BASE_PATH, STATE_PATH)
except Exception:
    online = None


def _safe_log_prediction(features: dict, result: dict) -> None:
    """Log a prediction if ONLINE_PRED_LOG is set. Never raises — logging must
    not break serving. Falls back to a no-op if auto_expert isn't importable
    (e.g. a slim inference-only container)."""
    if not PRED_LOG:
        return
    try:
        from auto_expert.logging import log_prediction
        log_prediction(
            log_path=PRED_LOG,
            request_id=result.get("prediction_id"),
            features=features,
            prediction=result.get("energy_wh"),
            model_version=result.get("model_version"),
            expert=result.get("expert"),
            gate_weights=result.get("gate_weights"),
        )
    except Exception:
        pass


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
            info = approved.info()
            if online is not None:
                info["online"] = online.info()
            self._send(200, info)
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/predict":
                feats = data.get("features", data)
                result = approved.predict(feats)
                _safe_log_prediction(feats, result)
                self._send(200, result)
            elif self.path == "/update":
                if online is None:
                    self._send(503, {"error": "online state unavailable"})
                    return
                self._send(200, online.update(
                    true_energy_wh=data["true_energy_wh"],
                    prediction_id=data.get("prediction_id"),
                    features=data.get("features"),
                    expert=data.get("expert"),
                ))
            elif self.path == "/predict_then_update":
                if online is None:
                    self._send(503, {"error": "online state unavailable"})
                    return
                self._send(200, online.predict_then_update(
                    data.get("features", {}), data["true_energy_wh"]))
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except KeyError as exc:
            self._send(400, {"error": f"missing field: {exc}"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})


def main():
    info = approved.info()
    print(f"Online service — model_kind={info['model_kind']}, "
          f"version={info['model_version']}, gate_mode={info['gate_mode']}")
    print(f"  served_from={info['model_path']}")
    if online is not None:
        oinfo = online.info()
        print(f"  online_state={STATE_PATH} loaded_from={oinfo['loaded_from']} "
              f"num_updates={oinfo['num_updates']}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving on 0.0.0.0:{PORT} "
          f"(GET /health /info ; POST /predict /update /predict_then_update)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

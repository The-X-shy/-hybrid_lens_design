"""
Structured stage logger for training/evaluation.
训练阶段结构化日志工具
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class StageLogger:
    """
    Write stage metrics/events into JSONL and summary JSON.
    将阶段指标/事件写入 JSONL 和 summary JSON
    """

    def __init__(
        self,
        result_dir: str,
        stage: str,
        mode: Optional[str] = None,
        wavelength: Optional[Any] = None,
    ):
        self.result_dir = Path(result_dir)
        self.stage = stage
        self.mode = mode
        self.wavelength = wavelength

        self.log_dir = self.result_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.summary_path = self.log_dir / "summary.json"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _base_payload(self) -> Dict[str, Any]:
        payload = {
            "timestamp": self._now(),
            "stage": self.stage,
        }
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.wavelength is not None:
            payload["wavelength"] = self.wavelength
        return payload

    def log_metric(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = self._base_payload()
        payload["type"] = "metric"
        if step is not None:
            payload["step"] = int(step)
        if epoch is not None:
            payload["epoch"] = int(epoch)
        payload["metrics"] = metrics or {}
        if lr is not None:
            payload["lr"] = lr
        if extra:
            payload.update(extra)
        self._write_jsonl(payload)

    def log_event(
        self,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._base_payload()
        record["type"] = "event"
        record["event"] = event
        if payload:
            record.update(payload)
        self._write_jsonl(record)

    def flush_summary(self, summary: Dict[str, Any]) -> None:
        data = self._base_payload()
        data["type"] = "summary"
        data.update(summary)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

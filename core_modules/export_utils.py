import json
import csv
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class Exporter:
    @staticmethod
    def export_to_json(report: Dict[str, Any], filepath: str):
        """Export the full pipeline trace and outputs to a JSON file."""
        def safe_serialize(obj):
            try:
                return json.loads(json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o))))
            except Exception as e:
                return {"serialization_error": str(e), "repr": str(obj)}
                
        serialized_report = {
            "speech": report.get("speech"),
            "environment": report.get("environment"),
            "situation": report.get("situation"),
            "world_state": safe_serialize(report.get("world_state")),
            "active_hyps": safe_serialize(report.get("active_hyps")),
            "trace": safe_serialize(report.get("trace")),
            "latencies": report.get("latencies")
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialized_report, f, indent=2)
        logger.info(f"Successfully exported trace to JSON: {filepath}")
            
    @staticmethod
    def export_to_csv(reports: List[Dict[str, Any]], filepath: str):
        """Export a list of summary reports to a CSV file."""
        if not reports:
            return
            
        keys = ["filename", "latency_ms", "speech", "environment", "situation", "dominant_state"]
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in reports:
                writer.writerow({
                    "filename": r.get("filename", "unknown"),
                    "latency_ms": sum(r.get("latencies", {}).values()),
                    "speech": r.get("speech", ""),
                    "environment": r.get("environment", ""),
                    "situation": r.get("situation", ""),
                    "dominant_state": getattr(r.get("world_state"), "dominant_state", "Unknown") if r.get("world_state") else "Unknown"
                })
        logger.info(f"Successfully exported summary to CSV: {filepath}")

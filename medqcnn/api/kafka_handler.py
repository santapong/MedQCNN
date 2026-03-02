"""
Kafka message handler for async inference pipeline.

Provides producer/consumer wrappers for event-driven medical image
inference. The inference results are published to an output topic
for downstream LLM agents (CaaS-Q orchestration).

Topics:
  - medqcnn.inference.request  — incoming image inference requests
  - medqcnn.inference.result   — outgoing diagnosis results

Note: Requires a running Kafka broker (see docker-compose.yml).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("medqcnn.kafka")

# Kafka connection defaults
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
REQUEST_TOPIC = "medqcnn.inference.request"
RESULT_TOPIC = "medqcnn.inference.result"


class KafkaInferenceProducer:
    """Publishes inference requests to Kafka.

    Used by the Agentic AI network to submit medical images
    for async quantum-classical inference.

    Args:
        bootstrap_servers: Kafka broker address(es).
        topic: Target topic for inference requests.
    """

    def __init__(
        self,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
        topic: str = REQUEST_TOPIC,
    ) -> None:
        self.topic = topic
        try:
            from kafka import KafkaProducer

            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info(f"Kafka producer connected to {bootstrap_servers}")
        except ImportError:
            logger.warning("kafka-python not installed. Run: uv add kafka-python")
            self.producer = None

    def send_request(
        self, image_path: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Send an inference request to Kafka."""
        if self.producer is None:
            logger.error("Kafka producer not available")
            return

        message = {
            "request_id": request_id,
            "image_path": image_path,
            "metadata": metadata or {},
        }
        self.producer.send(self.topic, value=message)
        self.producer.flush()
        logger.info(f"Sent request {request_id} to {self.topic}")


class KafkaInferenceConsumer:
    """Consumes inference requests and publishes results.

    Runs as a worker that:
    1. Reads image paths from the request topic
    2. Runs HybridQCNN inference
    3. Publishes diagnosis results to the result topic

    Args:
        bootstrap_servers: Kafka broker address(es).
        group_id: Consumer group ID.
    """

    def __init__(
        self,
        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
        group_id: str = "medqcnn-inference",
    ) -> None:
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self._consumer = None
        self._producer = None

    def _connect(self) -> bool:
        """Connect to Kafka. Returns True if successful."""
        try:
            from kafka import KafkaConsumer, KafkaProducer

            self._consumer = KafkaConsumer(
                REQUEST_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
            )
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            return True
        except ImportError:
            logger.warning("kafka-python not installed")
            return False
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            return False

    def run(self) -> None:
        """Start the inference worker loop."""
        if not self._connect():
            logger.error("Cannot start worker — Kafka not available")
            return

        from medqcnn.mcp.server import _ensure_model

        _ensure_model()  # pre-load model
        logger.info("Kafka inference worker started")

        for message in self._consumer:
            request = message.value
            request_id = request.get("request_id", "unknown")
            image_path = request.get("image_path", "")

            logger.info(f"Processing request {request_id}: {image_path}")

            try:
                # Use the MCP diagnose tool for inference
                from medqcnn.mcp.server import diagnose

                result_json = diagnose(image_path)
                result = json.loads(result_json)
                result["request_id"] = request_id
                result["status"] = "success"
            except Exception as e:
                result = {
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e),
                }

            self._producer.send(RESULT_TOPIC, value=result)
            self._producer.flush()
            logger.info(f"Published result for {request_id}")

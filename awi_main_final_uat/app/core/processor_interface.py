import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool = False
    processor_type: str = ""
    matches_found: int = 0
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None

class BaseProcessor:
    def __init__(self, processor_type: str, config: Dict[str, Any]):
        self.processor_type = processor_type
        self.config = config
        self.initialized = False

    def initialize(self, **kwargs) -> bool:
        raise NotImplementedError

    def process(self, document, **kwargs) -> ProcessingResult:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass

    def is_initialized(self) -> bool:
        return self.initialized

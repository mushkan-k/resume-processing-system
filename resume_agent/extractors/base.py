from abc import ABC, abstractmethod
from typing import Dict


class ResumeExtractor(ABC):
    """
    Contract for resume extraction implementations.
    """

    @abstractmethod
    def extract(self, resume_text: str) -> Dict:
        """
        Extract structured information from resume text.
        """
        pass

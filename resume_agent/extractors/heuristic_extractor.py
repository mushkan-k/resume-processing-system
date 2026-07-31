import re
from typing import Dict, List, Any

from .base import ResumeExtractor


class HeuristicResumeExtractor(ResumeExtractor):
    """
    Section-based + global-fallback deterministic resume extractor.

    Design goals:
    - Prefer section-based extraction
    - Fallback to global scan if sections are missing
    - Completeness over intelligence
    - Never return dummy or random data
    """

    SECTION_HEADERS = {
        "skills": [
            "skills",
            "technical skills",
            "key skills",
        ],
        "education": [
            "education",
            "academic background",
            "academics",
            "qualifications",
        ],
        "experience": [
            "experience",
            "work experience",
            "professional experience",
            "employment",
        ],
    }

    # ---------- public API ----------

    def extract(self, resume_text: str) -> Dict[str, Any]:
        sections = self._split_into_sections(resume_text)

        skills = self._extract_skills(
            sections.get("skills", ""),
            resume_text,
        )

        education = self._extract_education(
            sections.get("education", ""),
            resume_text,
        )

        experience = self._extract_experience(
            sections.get("experience", ""),
            resume_text,
        )

        return {
            "name": self._extract_name(resume_text),
            "email": self._extract_email(resume_text),
            "phone": self._extract_phone(resume_text),
            "skills": skills,
            "education": education,
            "experience": experience,
            "summary": self._build_summary(skills, experience),
            "score": self._compute_score(skills, education, experience),
        }

    # ---------- section handling ----------

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        lines = [l.rstrip() for l in text.splitlines()]
        sections: Dict[str, List[str]] = {}
        current_section = None

        for line in lines:
            clean = line.strip()
            if not clean:
                continue

            header = self._match_section_header(clean)
            if header:
                current_section = header
                sections.setdefault(current_section, [])
                continue

            if current_section:
                sections[current_section].append(clean)

        return {
            key: "\n".join(value)
            for key, value in sections.items()
        }

    def _match_section_header(self, line: str) -> str | None:
        lower = line.lower().strip(": ")

        for section, headers in self.SECTION_HEADERS.items():
            if lower in headers:
                return section

        return None

    # ---------- extraction logic ----------

    def _extract_skills(self, section_text: str, full_text: str) -> List[str]:
        """
        1. Prefer skills section
        2. Fallback: scan entire resume
        """
        source = section_text if section_text.strip() else full_text

        raw = re.split(r"[•,\n|\-:/]", source)
        skills = []

        for token in raw:
            clean = token.strip()
            if (
                1 < len(clean) <= 40
                and any(c.isalpha() for c in clean)
                and not clean.lower().startswith(
                    ("experience", "education", "project", "summary")
                )
            ):
                skills.append(clean)

        return sorted(set(skills))

    def _extract_education(
        self, section_text: str, full_text: str
    ) -> List[Dict[str, Any]]:
        """
        Prefer education section.
        Fallback: global scan for degree-like lines.
        """
        lines = (
            section_text.splitlines()
            if section_text.strip()
            else full_text.splitlines()
        )

        education = []
        for line in lines:
            clean = line.strip()
            if clean and any(
                kw in clean.lower()
                for kw in ["b.tech", "b.e", "bachelor", "master", "m.tech", "degree", "university"]
            ):
                education.append({"detail": clean})

        return education

    def _extract_experience(
        self, section_text: str, full_text: str
    ) -> List[Dict[str, Any]]:
        """
        Prefer experience section.
        Fallback: global scan for role / project indicators.
        """
        lines = (
            section_text.splitlines()
            if section_text.strip()
            else full_text.splitlines()
        )

        experience = []
        for line in lines:
            clean = line.strip()
            if clean and any(
                kw in clean.lower()
                for kw in [
                    "developer",
                    "engineer",
                    "intern",
                    "experience",
                    "project",
                    "worked",
                    "role",
                ]
            ):
                experience.append({"detail": clean})

        return experience

    # ---------- personal info ----------

    def _extract_name(self, text: str) -> str:
        for line in text.splitlines()[:10]:
            clean = line.strip()
            if (
                clean
                and clean.replace(" ", "").isalpha()
                and 1 < len(clean.split()) <= 4
            ):
                return clean.title()
        return ""

    def _extract_email(self, text: str) -> str:
        match = re.search(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}",
            text,
        )
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = re.search(
            r"(\+91[-\s]?)?[6-9]\d{9}",
            text,
        )
        return match.group(0) if match else ""

    # ---------- helpers ----------

    def _build_summary(
        self, skills: List[str], experience: List[Dict[str, Any]]
    ) -> str:
        if skills:
            return (
                f"Candidate has skills in {', '.join(skills[:5])} "
                f"and {len(experience)} experience entries."
            )
        return "Resume parsed using heuristic extraction."

    def _compute_score(
        self,
        skills: List[str],
        education: List[Dict[str, Any]],
        experience: List[Dict[str, Any]],
    ) -> float:
        score = 0.0
        score += min(len(skills), 20) * 2
        score += min(len(experience), 10) * 3
        score += min(len(education), 5) * 2
        return min(score, 100.0)

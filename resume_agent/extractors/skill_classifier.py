"""
Skill Classifier for post-extraction classification.
Called after resume extraction to classify each skill as PRIMARY or SECONDARY
based on the extracted experience context.
"""
import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are an expert HR analyst. Given an employee's work experience and their skill list, classify EACH skill as PRIMARY or SECONDARY.

**PRIMARY**: Core professional identity skill — used extensively across multiple roles, or the central focus of their career. This person could interview others on this skill.
**SECONDARY**: Supporting skill — used tangentially, in one brief role, or as minor support for primary work.

Rules:
- Typically 3-6 PRIMARY skills, rest are SECONDARY
- 15+ years or central to most roles → PRIMARY
- Only 1 short role with no depth → SECONDARY
- Soft skills are SECONDARY unless the role is explicitly about them
- When in doubt → SECONDARY (PRIMARY means interview-panel-worthy)

## Experience:
{experience_text}

## Skills to Classify:
{skills_list}

Return ONLY a JSON object with this exact structure:
{{"classifications": [{{"skill": "name", "type": "PRIMARY", "reason": "one sentence"}}]}}

Classify EVERY skill listed. Return nothing else.
"""


class SkillClassifier:
    """Classifies skills as PRIMARY/SECONDARY using Azure OpenAI GPT-4o."""
    
    def __init__(self):
        from openai import AzureOpenAI
        
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT not set")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    
    def classify(
        self,
        skills: List[str],
        experience: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Classify a list of skills based on experience context.
        
        Args:
            skills: List of skill names
            experience: List of experience dicts with role/company/duration/description
            
        Returns:
            List of {"skill": str, "type": "PRIMARY"|"SECONDARY", "reason": str}
        """
        if not skills:
            return []
        
        # Format experience for prompt
        exp_lines = []
        for i, exp in enumerate(experience, 1):
            role = exp.get("role", "Unknown")
            company = exp.get("company", "Unknown")
            duration = exp.get("duration", "")
            description = exp.get("description", "")
            exp_lines.append(f"{i}. {role} at {company} ({duration})")
            if description:
                exp_lines.append(f"   {description}")
        
        experience_text = "\n".join(exp_lines) if exp_lines else "No experience data."
        skills_list = "\n".join(f"- {s}" for s in skills)
        
        prompt = CLASSIFICATION_PROMPT.format(
            experience_text=experience_text,
            skills_list=skills_list
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise HR classification engine. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            
            # Handle wrapper key
            if "classifications" in parsed:
                results = parsed["classifications"]
            elif isinstance(parsed, list):
                results = parsed
            else:
                # Find any list in the response
                for val in parsed.values():
                    if isinstance(val, list):
                        results = val
                        break
                else:
                    logger.error(f"Unexpected response structure: {parsed.keys()}")
                    return self._default_classifications(skills)
            
            # Validate & normalize
            classified = []
            seen = set()
            for item in results:
                skill = item.get("skill", "").strip()
                skill_type = item.get("type", "SECONDARY").upper().strip()
                reason = item.get("reason", "")
                
                if skill_type not in ("PRIMARY", "SECONDARY"):
                    skill_type = "SECONDARY"
                
                if skill:
                    classified.append({
                        "skill": skill,
                        "type": skill_type,
                        "reason": reason
                    })
                    seen.add(skill.lower())
            
            # Catch any missed skills
            for s in skills:
                if s.lower() not in seen:
                    classified.append({
                        "skill": s,
                        "type": "SECONDARY",
                        "reason": "Not classified by LLM — defaulted"
                    })
            
            return classified
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._default_classifications(skills)
    
    def _default_classifications(self, skills: List[str]) -> List[Dict[str, str]]:
        """Fallback: mark all as unclassified/SECONDARY."""
        return [
            {"skill": s, "type": "SECONDARY", "reason": "Classification unavailable"}
            for s in skills
        ]

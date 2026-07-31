"""OpenAI Extractor - PRODUCTION READY"""
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class EducationItem(BaseModel):
    degree: str = Field(default="")
    institution: str = Field(default="")
    year: str = Field(default="")


class ExperienceItem(BaseModel):
    role: str = Field(default="")
    company: str = Field(default="")
    duration: str = Field(default="")
    description: str = Field(default="")


class ResumeSchema(BaseModel):
    name: str = Field(default="")
    email: str = Field(default="")
    phone: str = Field(default="")
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    summary: str = Field(default="")


class OpenAIResumeExtractor:
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        if api_key and endpoint and deployment:
            self.llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=deployment,
                temperature=0,
            )
        else:
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_KEY not set")

            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")),
                temperature=0,
                api_key=api_key
            )
        
        self.parser = PydanticOutputParser(pydantic_object=ResumeSchema)
        
        self.prompt = PromptTemplate(
            input_variables=["resume_text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template="""
Extract ALL information from this resume.

{format_instructions}

Rules:
- Extract ALL skills (technical, tools, frameworks)
- Extract ALL education entries
- Extract ALL experience entries
- If missing, return empty string or empty list
- Be thorough and complete

Resume:
{resume_text}
""".strip()
        )
    
    def extract(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured data from resume"""
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({"resume_text": resume_text})
            
            extracted = result.dict()
            extracted["score"] = self._compute_score(extracted)
            
            return extracted
            
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
            # Return minimal valid structure
            return {
                "name": "",
                "email": "",
                "phone": "",
                "skills": [],
                "education": [],
                "experience": [],
                "summary": "Extraction failed",
                "score": 0.0
            }
    
    def _compute_score(self, extracted: Dict[str, Any]) -> float:
        """Compute resume quality score"""
        score = 10.0  # Base score
        
        skills = extracted.get("skills", [])
        score += min(len(skills) * 2, 40)
        
        experience = extracted.get("experience", [])
        score += min(len(experience) * 5, 30)
        
        education = extracted.get("education", [])
        score += min(len(education) * 5, 15)
        
        if extracted.get("name"):
            score += 5
        if extracted.get("email"):
            score += 5
        if extracted.get("phone"):
            score += 5
        
        return min(score, 100.0)
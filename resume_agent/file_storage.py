"""File storage configuration"""
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


class FileStorageConfig:
    BASE_DIR = Path(os.getenv("FILE_STORAGE_BASE", "C:/data/resume_system"))
    RESUMES_DIR = BASE_DIR / "resumes"
    
    @classmethod
    def initialize_directories(cls):
        """Create directories"""
        cls.RESUMES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[FILE] Directories ready: {cls.BASE_DIR}")
    
    @classmethod
    def get_resume_path(cls, resume_id, file_type):
        """Get path for original resume file"""
        file_type = file_type.lower().lstrip('.')
        return cls.RESUMES_DIR / f"{resume_id}.{file_type}"
    
    @classmethod
    def save_base64_to_file(cls, base64_content, file_path):
        """Decode and save file"""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        decoded = base64.b64decode(base64_content)
        file_path.write_bytes(decoded)
        
        print(f"[FILE] Saved: {file_path} ({len(decoded)} bytes)")
        return str(file_path.absolute())
import os
import requests
import logging
from typing import Any, Dict, List
from time import sleep
from datetime import datetime


class JobDivaClient:
    """
    JobDiva API client with SAFETY LIMITS to prevent runaway API calls.
    """
    
    BASE_URL = "https://api.jobdiva.com/apiv2"
    RETRY_LIMIT = 3
    RETRY_DELAY = 5
    
    #  SAFETY LIMIT: Maximum candidates to process in one run
    MAX_CANDIDATES_PER_RUN = 10  # Start with 10 for testing
    
    def __init__(self):
        self.client_id = os.getenv("JOBDIVA_CLIENT_ID")
        self.username = os.getenv("JOBDIVA_USERNAME")
        self.password = os.getenv("JOBDIVA_PASSWORD")

        if not all([self.client_id, self.username, self.password]):
            raise RuntimeError("JobDiva credentials not set")

        self._token = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # ---------------- AUTH ----------------

    def authenticate(self) -> str:
        """Authenticate with JobDiva and get JWT token."""
        url = f"{self.BASE_URL}/authenticate"
        params = {
            "clientid": self.client_id,
            "username": self.username,
            "password": self.password,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        token = response.text.strip()
        if token.count(".") != 2:
            raise RuntimeError("Invalid JWT token from JobDiva")

        self._token = token
        self.logger.info(" JobDiva authentication successful")
        return token

    def _headers(self):
        """Get authorization headers."""
        if not self._token:
            self.authenticate()
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _get(self, url, params=None):
        """Make GET request with retry logic."""
        for i in range(self.RETRY_LIMIT):
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r
            except Exception as e:
                if i == self.RETRY_LIMIT - 1:
                    raise
                self.logger.warning(f"Request failed, retrying... ({i+1}/{self.RETRY_LIMIT})")
                sleep(self.RETRY_DELAY)

    # ---------------- DATE CONVERSION ----------------

    @staticmethod
    def _convert_date_format(date_str: str) -> str:
        """
        Convert YYYY-MM-DD to JobDiva format: MM/DD/YYYY HH:MM:SS
        """
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y 00:00:00")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    # ---------------- MAIN METHOD ----------------

    def get_resumes(self, from_date: str, to_date: str, max_candidates: int = None) -> List[Dict[str, str]]:
        """
        Get resumes for candidates in date range.
        
        Args:
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            max_candidates: Maximum candidates to process (default: MAX_CANDIDATES_PER_RUN)
            
        Returns:
            List of dicts with candidate_id and resume_id
        """
        if max_candidates is None:
            max_candidates = self.MAX_CANDIDATES_PER_RUN
        
        self.logger.info(f" Fetching resumes from {from_date} to {to_date}")
        self.logger.info(f"  SAFETY LIMIT: Will process maximum {max_candidates} candidates")
        
        # Convert dates
        from_date_jd = self._convert_date_format(from_date)
        to_date_jd = self._convert_date_format(to_date)
        
        # Get candidates
        all_candidates = self._get_new_updated_candidates(from_date_jd, to_date_jd)
        self.logger.info(f" API returned {len(all_candidates)} total candidates")
        
        #  APPLY LIMIT
        if len(all_candidates) > max_candidates:
            self.logger.warning(
                f"  LIMITING to {max_candidates} candidates "
                f"(API returned {len(all_candidates)})"
            )
            candidates = all_candidates[:max_candidates]
        else:
            candidates = all_candidates
        
        self.logger.info(f" Processing {len(candidates)} candidates")
        
        # Get resume IDs
        resumes = []
        for i, candidate in enumerate(candidates, 1):
            candidate_id = candidate.get("CANDIDATEID")
            if not candidate_id:
                continue
            
            self.logger.info(f"[{i}/{len(candidates)}] Getting resumes for candidate {candidate_id}")
            
            try:
                candidate_resumes = self._get_candidate_resume_ids(str(candidate_id))
                resumes.extend(candidate_resumes)
            except Exception as e:
                self.logger.warning(f"  Failed candidate {candidate_id}: {e}")
                continue
        
        self.logger.info(f" Found {len(resumes)} total resumes")
        return resumes

    def _get_new_updated_candidates(self, from_date: str, to_date: str) -> List[Dict]:
        """Fetch candidates modified in date range."""
        url = f"{self.BASE_URL}/bi/NewUpdatedCandidateRecords"
        
        params = {
            "fromDate": from_date,
            "toDate": to_date,
        }
        
        self.logger.info(f"Calling NewUpdatedCandidateRecords: {from_date} → {to_date}")
        
        response = self._get(url, params=params)
        data = response.json()
        
        if "data" not in data:
            self.logger.warning("No 'data' field in response")
            return []
        
        return data["data"]

    def _get_candidate_resume_ids(self, candidate_id: str) -> List[Dict[str, str]]:
        """Get resume IDs for a candidate."""
        url = f"{self.BASE_URL}/bi/CandidateResumesDetail"
        
        params = {"candidateId": candidate_id}
        
        response = self._get(url, params=params)
        data = response.json()
        
        if "data" not in data or not data["data"]:
            return []
        
        resumes = []
        for resume_data in data["data"]:
            resume_id = resume_data.get("RESUMEID")
            if resume_id:
                resumes.append({
                    "candidate_id": str(candidate_id),
                    "resume_id": str(resume_id),
                })
        
        return resumes

    def get_resume_content(self, resume_id: str) -> Dict[str, Any]:
        """Fetch resume content (base64 and plaintext)."""
        url = f"{self.BASE_URL}/bi/ResumeDetail"
        
        params = {"resumeId": resume_id}
        
        response = self._get(url, params=params)
        data = response.json()
        
        if "data" not in data or not data["data"]:
            raise ValueError(f"No data returned for resume {resume_id}")
        
        resume_data = data["data"][0]
        
        return {
            "base64": resume_data.get("FILECONTENT_BASE64ENCODED", ""),
            "text": resume_data.get("PLAINTEXT", ""),
        }
# Resume Processing System

A resume ingestion pipeline that fetches resumes from JobDiva, stores them in MySQL, and extracts structured data using OpenAI via MCP.

---

## What This Does

- Fetches resumes from JobDiva
- Stores resume base64 in MySQL 
- Saves resume files locally
- Extracts structured data using OpenAI
- Stores extracted data and normalized skills in MySQL

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- JobDiva API credentials
- OpenAI API key

---

## Setup & Run

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/resume-mcp-system.git
cd resume-mcp-system
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate:**

Windows:
```bash
venv\Scripts\activate
```

macOS / Linux:
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3305
MYSQL_DATABASE=resume_processing
MYSQL_USER=resume_user
MYSQL_PASSWORD=resume_password

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

JOBDIVA_CLIENT_ID=your_client_id
JOBDIVA_USERNAME=your_username
JOBDIVA_PASSWORD=your_password
```

---

### 5. Setup Database (One Command)

```powershell
.\DEPLOY.ps1
```

Starts MySQL, applies schema, and verifies DB connection.

---

## Run Commands

### Single Resume (Recommended First)

Edit IDs inside `test_single_resume.py`, then run:

```bash
python test_single_resume.py
```

---

### Batch Processing (Safe Mode)

```bash
python test_safe_batch.py
```

Default safety limit:
```python
MAX_CANDIDATES_PER_RUN = 10
```

---

### Verify Data

```bash
python scripts/verify_db.py
```

Expected: records exist in `resumes`, `extracted_resume_data`, `resume_skills`.

---

## Architecture Flow

```
JobDiva API
    ↓
1. Fetch Resume (base64 + text)
    ↓
2. Save to MySQL (resumes table)
    ↓
3. Save Original File (PDF/DOC/DOCX)
    ↓
4. Create Temp File from DB
    ↓
5. OpenAI Extraction
    ↓
6. Save Extracted Data (MySQL)
    ↓
7. Save Skills (MySQL)
```


---

## Project Structure

```
resume-mcp-system/
├── resume_agent/
│   ├── agent.py                 # Main orchestration
│   ├── file_storage.py          # File handling
│   ├── integrations/
│   │   └── jobdiva_client.py    # JobDiva API
│   └── extractors/
│       └── openai_extractor.py  # OpenAI extraction
│
├── mcp_server/
│   ├── server.py                # MCP server
│   ├── config.py                # Environment config
│   └── db/
│       ├── connection.py        # MySQL connection
│       └── models.py            # Database writes
│
├── scripts/
│   └── verify_db.py             # DB verification
│
├── schema.sql                   # Database schema
├── docker-compose.yml           # MySQL container
├── DEPLOY.ps1                   # DB setup script
├── test_single_resume.py        # Single resume test
├── test_safe_batch.py           # Batch processing
└── .env.example                 # Environment template
```

---

## File Storage

Resume files are stored at:

```
C:\data\resume_system\
├── resumes\   # Original resumes
└── temp\      # Temporary extraction files
```

---

## Tech Stack

- Python 3.11
- MySQL 8 (Docker)
- JobDiva API
- OpenAI API
- MCP (Model Context Protocol)

---

## Status

- Single resume ingestion: Working
- Batch ingestion: Working


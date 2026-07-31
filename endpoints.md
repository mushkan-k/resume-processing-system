## JobDiva API Reference (External)

**Base URL:** `https://api.jobdiva.com/apiv2`

**Authentication:**
All endpoints require a Bearer token obtained from:
```
GET /authenticate?clientid={JOBDIVA_CLIENT_ID}&username={JOBDIVA_USERNAME}&password={JOBDIVA_PASSWORD}
```
Headers for all subsequent calls:
```
Authorization: Bearer {token}
Accept: application/json
```

---

### 1. Authentication
```
GET /apiv2/authenticate?clientid={JOBDIVA_CLIENT_ID}&username={JOBDIVA_USERNAME}&password={JOBDIVA_PASSWORD}
```
**Returns:** Plain text token (string, ~1108 chars)

---

### 2. New/Updated Job Records
```
GET /apiv2/bi/NewUpdatedJobRecords
```
**Params:**
| Parameter | Required | Format | Notes |
|-----------|----------|--------|-------|
| `fromDate` | Yes | `MM/DD/YYYY HH:MM:SS` | Max 14-day window |
| `toDate` | Yes | `MM/DD/YYYY HH:MM:SS` | |
| `jobStatus` | No | `OPEN`, `CLOSED`, etc. | Filter by status |

**Response:** `{ "message": "...", "data": [ { JOBID, JOBDIVANO, DIVISIONNAME, COMPANYNAME, ... } ] }`

**Key fields:** `JOBID`, `JOBDIVANO`, `OPTIONALREFERENCENO`, `DIVISIONID`, `DIVISIONNAME`, `UPDATEDBY`, `PRIMARYRECRUITERID`, `PRIMARYSALESID`, `MSPNAME`, `MSPID`, `BILLRATE_CURRENCY`, `BLS_OCCUPATION_ID`, `BLS_OCCUPATION`, `POST_UPDATED`, `ONSITE_REMOTE`

---

### 3. Job Applicants Detail
```
GET /apiv2/bi/JobApplicantsDetail
```
**Params:**
| Parameter | Required | Format | Notes |
|-----------|----------|--------|-------|
| `jobId` | Yes | integer | JOBID from NewUpdatedJobRecords |

**Response:**
```json
{
  "message": "Query \"JobApplicantsDetail\" completed successfully",
  "data": [
    {
      "CANDIDATEID": "2504749350992",
      "FIRSTNAME": "Rajesh",
      "LASTNAME": "Deshmukh",
      "EMAIL": "raj.deshmukh05@gmail.com",
      "JOBID": "28588667",
      "DATEAPPLIED": "2026-06-11T13:26:01",
      "RESUMESOURCE": "LinkedIn Apply Connect",
      "ACTION": "",
      "ACTIONUSERID": "",
      "REJECTREASON": "",
      "ACTIONDATE": "",
      "REFERRER": "",
      "RESUMEID": "2504749350992_1013_1",
      "STATUS": "Pending"
    }
  ]
}
```

**Linked to:** Uses `JOBID` from `NewUpdatedJobRecords`. Returns `RESUMEID` usable in `ResumeDetail`.

---

### 4. New/Updated Candidate Records (Full Candidate DB)
```
GET /apiv2/bi/NewUpdatedCandidateRecords
```
**Purpose:** Fetch all candidates created/updated in a date range from JobDiva's entire candidate database (not just your company's employees — this is the full shared pool across all JobDiva clients).

**Params:**
| Parameter | Required | Format | Notes |
|-----------|----------|--------|-------|
| `fromDate` | Yes | `MM/DD/YYYY HH:MM:SS` | URL-encoded (e.g., `01%2F21%2F2026%2013%3A00%3A00`) |
| `toDate` | Yes | `MM/DD/YYYY HH:MM:SS` | URL-encoded |

**Example:**
```
GET /apiv2/bi/NewUpdatedCandidateRecords?fromDate=01%2F21%2F2026%2013%3A00%3A00&toDate=01%2F21%2F2026%2013%3A05%3A00
```

**Response:**
```json
{
  "message": "Query \"NewUpdatedCandidateRecords\" completed successfully",
  "data": [
    {
      "CANDIDATEID": "19947961725624",
      "FIRSTNAME": "John",
      "LASTNAME": "Smith",
      "EMAIL": "john.smith@email.com",
      "CITY": "Dallas",
      "STATE": "TX",
      "COUNTRY": "US",
      "PHONE": "555-123-4567",
      "CREATEDDATE": "2026-01-21T13:02:15",
      "UPDATEDDATE": "2026-01-21T13:02:15"
    }
  ]
}
```

**Key fields:** `CANDIDATEID` (used in next step), `FIRSTNAME`, `LASTNAME`, `EMAIL`, `CITY`, `STATE`, `COUNTRY`

---

### 5. Candidate Resumes Detail
```
GET /apiv2/bi/CandidateResumesDetail
```
**Purpose:** Get all resume IDs/versions for a specific candidate.

**Params:**
| Parameter | Required | Format | Notes |
|-----------|----------|--------|-------|
| `candidateId` | Yes | string | CANDIDATEID from step 4 or from JobApplicantsDetail |

**Example:**
```
GET /apiv2/bi/CandidateResumesDetail?candidateId=19947961725624
```

**Response:**
```json
{
  "message": "Query \"CandidateResumesDetail\" completed successfully",
  "data": [
    {
      "RESUMEID": "19947961725624_1013_1",
      "FILENAME": "John_Smith_Resume.pdf",
      "UPLOADDATE": "2026-01-21T13:02:15",
      "RESUMETITLE": "Software Engineer Resume"
    }
  ]
}
```

**Key fields:** `RESUMEID` (used in next step), `FILENAME`, `UPLOADDATE`

---

### 6. Resume Detail (Base64 file)
```
GET /apiv2/bi/ResumeDetail
```
**Purpose:** Download the actual resume file as base64-encoded content.

**Params:**
| Parameter | Required | Format | Notes |
|-----------|----------|--------|-------|
| `resumeId` | Yes | string | RESUMEID from CandidateResumesDetail or JobApplicantsDetail |

**Example:**
```
GET /apiv2/bi/ResumeDetail?resumeId=19947961725624_1013_1
```

**Response:**
```json
{
  "message": "Query \"ResumeDetail\" completed successfully",
  "data": [
    {
      "FILENAME": "John_Smith_Resume.pdf",
      "FILECONTENT_BASE64ENCODED": "JVBERi0xLjQKJe...",
      "PLAINTEXT": "JOHN SMITH\nSoftware Engineer\nDallas, TX..."
    }
  ]
}
```

**Key fields:** `FILECONTENT_BASE64ENCODED` (base64 encoded — decode to get PDF/DOC file), `FILENAME`, `PLAINTEXT` (parsed plain text of resume)

**Decoding example (Python):**
```python
import base64
content = base64.b64decode(data["FILECONTENT_BASE64ENCODED"])
with open(data["FILENAME"], "wb") as f:
    f.write(content)
```

---

### API Chain (how they link together)
```
NewUpdatedJobRecords ──(JOBID)──► JobApplicantsDetail ──(RESUMEID)──► ResumeDetail
                                        │
                                        └──(CANDIDATEID)──► CandidateResumesDetail ──(RESUMEID)──► ResumeDetail

NewUpdatedCandidateRecords ──(CANDIDATEID)──► CandidateResumesDetail ──(RESUMEID)──► ResumeDetail
```

---

## Our API Endpoints

### Job Descriptions
GET /api/jds
  Query Params:
    - from_date (string, MM/DD/YYYY HH:MM:SS) — default: last 14 days (JobDiva max window is 14 days, auto-chunked for larger ranges)
    - to_date (string, MM/DD/YYYY HH:MM:SS) — default: now
    - job_status (string) — e.g. "OPEN", "CLOSED"
    - search (string) — searches title, companyName, city
    - page (int, default 1)
    - page_size (int, default 100, max 500)

  Response:
  {
    "data": [ { jobId, jobDivaNo, divisionName, companyName, issueDate, startDate, endDate, positionType, jobStatus, title, maxAllowedSubmittals, address1, address2, city, state, zipCode, country } ],
    "total": 150
  }

### Job Description Detail
GET /api/jds/{jobId}

  Example: GET /api/jds/28116432

  Response:
  {
    "jobId": "28116432",
    "jobDivaNo": "26-02701",
    "title": "Marketing Analyst - II",
    "companyName": "Verizon One",
    "divisionName": "Talent-on-Demand",
    "city": "Basking Ridge",
    "state": "NJ",
    "zipCode": "07920",
    "country": "US",
    "address1": "One Verizon Way",
    "address2": "~ ~",
    "positionType": "Contract",
    "jobStatus": "ON HOLD",
    "issueDate": "2026-05-01T08:35:38",
    "startDate": "2026-06-01T00:00:00",
    "endDate": "2027-06-04T00:00:00",
    "description": "Full job description HTML...",
    "requiredSkills": [],
    "openings": 1,
    "positions": 1,
    "fills": 0,
    "maxAllowedSubmittals": 2,
    "billRateMin": 0,
    "billRateMax": 77,
    "billRatePer": "Hour",
    "payRateMin": 45,
    "payRateMax": 50,
    "payRatePer": "Hour",
    "experienceLevel": null,
    "refNo": "VZGTJP00060650",
    "submittalDue": "2026-06-01T00:00:00",
    "priority": 1
  }

---

### Interviewer Panel Matching (NEW)

**GET /api/jds/interviewers/{jobId}**
Find the best internal employees to serve as interviewers for a JD based on PRIMARY skill overlap.

Query Params:
  - top_n (int, default 10, max 50) — Max interviewers to return
  - min_score (float, default 0.2) — Minimum match score threshold

Response:
```json
{
  "jd": { "title": "Software Engineer 3", "cluster": "IN | Software Engineer", "company": "IN Caterpillar" },
  "interviewers": [
    {
      "employeeId": 18668318985372,
      "employeeName": "Chenxi Gao",
      "score": 0.75,
      "primaryMatches": ["Python", "AWS", "Microservices"],
      "secondaryMatches": ["Docker"],
      "totalPrimarySkills": 5,
      "totalSecondarySkills": 8,
      "clientName": "Caterpillar",
      "deliveryCenter": "Onshore"
    }
  ],
  "jdSkills": ["Python", "AWS", "Microservices", "Docker", "CI/CD"],
  "totalCandidatesScored": 42
}
```

**GET /api/jds/employee-skills/{employeeId}**
Get full skill profile with PRIMARY/SECONDARY classification breakdown.

Response:
```json
{
  "employeeId": 18668318985372,
  "primary": [
    { "skill": "Python", "reason": "Used across 4 roles over 8 years as core development language", "classifiedAt": "2026-07-08T10:30:00" }
  ],
  "secondary": [
    { "skill": "Docker", "reason": "Used in one role for containerization support", "classifiedAt": "2026-07-08T10:30:00" }
  ],
  "unclassified": [],
  "summary": {
    "totalSkills": 14,
    "primaryCount": 5,
    "secondaryCount": 9,
    "unclassifiedCount": 0,
    "classificationComplete": true
  }
}
```

---

### Predictions (v4 Job Demand Forecasts)

**GET /api/predictions/summary**
KPI summary for dashboard header cards.

Response:
```json
{
  "forecastWindow": "2026-07 to 2026-12",
  "totalClusters": 35,
  "reliableClusters": 29,
  "totalDemand6Months": 1072,
  "averageMape": 49.7,
  "overallAccuracy": 50.3,
  "modelVersion": "v4_lstm_optimized",
  "topClusters": [
    {"cluster": "Engineering", "demand": 223},
    {"cluster": "Software Engineering", "demand": 96}
  ]
}
```

**GET /api/predictions/clusters**
All 35 clusters with accuracy grades and demand totals.

Response:
```json
{
  "clusters": [
    {
      "cluster": "Engineering",
      "model": "lstm_ensemble",
      "totalDemand": 223,
      "forecastRange": "2026-07 to 2026-12",
      "mape": 19.1,
      "accuracy": 80.9,
      "mase": 0.65,
      "grade": "A",
      "reliable": true,
      "topSkills": ["PYTHON","TESTING","AUTOMATION","MEDICAL DEVICE","PROJECT MANAGEMENT"]
    }
  ],
  "total": 35
}
```

**GET /api/predictions**
Quarterly demand overview (filterable).

Query Params:
  - quarter (string, optional) - e.g. "2026-Q3"
  - reliable_only (bool, default true)

**GET /api/predictions/cluster/{cluster_name}**
Monthly predictions for a specific cluster (for line/area charts).

Example: GET /api/predictions/cluster/Engineering

Response:
```json
{
  "cluster": "Engineering",
  "model": "lstm_ensemble",
  "mape": 19.1,
  "accuracy": 80.9,
  "reliable": true,
  "topSkills": ["PYTHON","TESTING","AUTOMATION","MEDICAL DEVICE","PROJECT MANAGEMENT"],
  "totalDemand": 223,
  "predictions": [
    {"month": "2026-07", "demand": 44, "lower": 23, "upper": 64},
    {"month": "2026-08", "demand": 41, "lower": 21, "upper": 63}
  ]
}
```

**GET /api/predictions/skills**
Skills ranked by weighted demand - tells recruiters what to prioritize.

Response:
```json
{
  "skills": [
    {"skill": "PYTHON", "weightedDemand": 449, "clusters": ["Engineering","Software Engineering","Quality Assurance","IT Support"]},
    {"skill": "PROJECT MANAGEMENT", "weightedDemand": 321, "clusters": ["Engineering","Project Management","Marketing"]}
  ],
  "total": 29
}
```


-- =============================================================
-- SCHEMA REFERENCE — Resume Processing System
-- =============================================================
-- Tables are created/modified MANUALLY inside the MySQL container
--
-- To connect:
--   docker exec -it resume_mysql mysql -u resume_user -presume_password resume_processing
--
-- No need to restart Docker for schema changes!
-- =============================================================

SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- =============================================================
-- Table 1: employee
-- Master data uploaded from Excel (JobDiva Active Consultants)
-- Data source: scripts/upload_employees.py
-- =============================================================
CREATE TABLE employee (
    id BIGINT PRIMARY KEY,                          -- CandidateID from JobDiva (e.g., 18668318985372)
    employee_name TEXT NOT NULL,                     -- Full name (e.g., "Chenxi Gao")
    job_diva_no TEXT NOT NULL,                       -- JobDiva number (e.g., "25-16373")
    delivery_center TEXT,                            -- e.g., "Onshore"
    division_name TEXT,                              -- e.g., "Talent-on-Demand"
    client_name TEXT,                                -- e.g., "Caterpillar"
    employee_id INT DEFAULT NULL,                   -- Internal employee ID (e.g., 9474)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- =============================================================
-- Table 2: resume
-- Resume data fetched from JobDiva API + extracted via OpenAI
-- Linked to employee via FK (employee_jobdiva_id → employee.id)
-- =============================================================
CREATE TABLE resume (
    resume_id TEXT NOT NULL,                        -- e.g., "18668318985372_1013_1"
    employee_jobdiva_id BIGINT NOT NULL,            -- FK → employee.id
    resume_base64 LONGTEXT,                         -- Raw base64 encoded file
    file_type TEXT,                                  -- "pdf", "docx", "doc"
    file_path TEXT,                                  -- Path on disk (e.g., "C:/data/resume_system/resumes/...")
    email TEXT,                                      -- Extracted by OpenAI
    phone TEXT,                                      -- Extracted by OpenAI
    education JSON,                                  -- Extracted by OpenAI (array of objects)
    experience JSON,                                 -- Extracted by OpenAI (array of objects)
    skills JSON,                                     -- Extracted by OpenAI (array of strings)
    summary TEXT,                                    -- Extracted by OpenAI
    score FLOAT DEFAULT 0,                          -- Resume quality score (computed)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (resume_id(255)),
    INDEX idx_employee_jobdiva_id (employee_jobdiva_id),

    FOREIGN KEY (employee_jobdiva_id) REFERENCES employee(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- =============================================================
-- Table 3: employee_skillset
-- Normalized skills — one row per skill per resume per employee
-- Composite PK: (resume_id, employee_jobdiva_id, skill)
-- =============================================================
CREATE TABLE employee_skillset (
    resume_id TEXT NOT NULL,                        -- FK-like → resume.resume_id
    employee_jobdiva_id BIGINT NOT NULL,            -- FK → employee.id
    skill VARCHAR(255) NOT NULL,                    -- e.g., "Python", "AWS", "Docker"
    skill_type ENUM('PRIMARY', 'SECONDARY') DEFAULT NULL,  -- Classification result
    classification_reason TEXT DEFAULT NULL,        -- GPT-4o reasoning for the classification
    classified_at TIMESTAMP DEFAULT NULL,           -- When classification was performed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (resume_id(255), employee_jobdiva_id, skill),
    INDEX idx_skill (skill),
    INDEX idx_skill_type (skill_type),
    INDEX idx_resume_id (resume_id(255)),
    INDEX idx_employee_jobdiva_id (employee_jobdiva_id),

    FOREIGN KEY (employee_jobdiva_id) REFERENCES employee(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- =============================================================
-- Table 4: skill_ranking_history
-- Tracks per-skill employee count over time (auto-recorded after pipeline runs)
-- Used for "View History" timeline when clicking a skill in the dashboard
-- =============================================================
CREATE TABLE skill_ranking_history (
    id INT AUTO_INCREMENT PRIMARY KEY,              -- Row ID
    skill VARCHAR(255) NOT NULL,                    -- e.g., "Python", "AWS"
    employee_count INT NOT NULL,                    -- Number of employees with this skill at that time
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- When this count was recorded
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- =============================================================
-- Table 5: skill_practice_mapping
-- Maps each skill to a practice (e.g., "Python" → "Engineering")
-- Practices: Engineering, Data & Analytics, Cloud & Infrastructure,
--   AI / Machine Learning, Cybersecurity, Project Management,
--   Quality Assurance, UI/UX & Frontend, ERP & Business Apps,
--   Database & Storage, Networking & Telecom, Business Analysis,
--   DevOps / SRE, Other
-- =============================================================
-- CREATE TABLE skill_practice_mapping (
--     skill TEXT NOT NULL,
--     practice TEXT NOT NULL,
--     PRIMARY KEY (skill(255)),
--     INDEX idx_practice (practice(100))
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""
Upload employee data from Excel to MySQL database.

Usage:
    python scripts/upload_employees.py

Excel Path: C:/Users/kmush/Downloads/JobDiva Active Consultants.xlsx
"""

import os
import sys
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
EXCEL_PATH = r"C:\Users\kmush\Downloads\JobDiva Active Consultants.xlsx"

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", 3305)),
    "database": os.getenv("MYSQL_DATABASE", "resume_processing"),
    "user": os.getenv("MYSQL_USER", "resume_user"),
    "password": os.getenv("MYSQL_PASSWORD", "resume_password"),
}

# Column mapping: Excel column -> DB column
COLUMN_MAP = {
    "CandidateID": "id",
    "Candidate": "employee_name",
    "JobdivaNO": "job_diva_no",
    "DeliveryCentre": "delivery_center",
    "DivisionName": "division_name",
    "ClientName": "client_name",
    "EmployeeID": "employee_id",
}


def upload_employees():
    # --- Step 1: Read Excel ---
    print(f"Reading Excel file: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Keep only the columns we need
    df = df[list(COLUMN_MAP.keys())]

    # Rename columns to match DB
    df = df.rename(columns=COLUMN_MAP)

    # Clean data
    df["employee_name"] = df["employee_name"].fillna("").astype(str).str.strip()
    df["job_diva_no"] = df["job_diva_no"].fillna("").astype(str).str.strip()
    df["delivery_center"] = df["delivery_center"].fillna("").astype(str).str.strip()
    df["division_name"] = df["division_name"].fillna("").astype(str).str.strip()
    df["client_name"] = df["client_name"].fillna("").astype(str).str.strip()

    # Handle employee_id: convert to int where possible, else None
    df["employee_id"] = pd.to_numeric(df["employee_id"], errors="coerce")
    df["employee_id"] = df["employee_id"].where(df["employee_id"].notna(), None)

    print(f"Total rows to upload: {len(df)}")
    print(f"Sample data:\n{df.head()}\n")

    # --- Step 2: Connect to MySQL ---
    print(f"Connecting to MySQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # --- Step 3: Insert data ---
    insert_query = """
        INSERT INTO employee (id, employee_name, job_diva_no, delivery_center, division_name, client_name, employee_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            employee_name = VALUES(employee_name),
            job_diva_no = VALUES(job_diva_no),
            delivery_center = VALUES(delivery_center),
            division_name = VALUES(division_name),
            client_name = VALUES(client_name),
            employee_id = VALUES(employee_id)
    """

    success = 0
    errors = 0

    for index, row in df.iterrows():
        try:
            emp_id = int(row["employee_id"]) if row["employee_id"] is not None and pd.notna(row["employee_id"]) else None

            values = (
                int(row["id"]),
                row["employee_name"],
                row["job_diva_no"],
                row["delivery_center"],
                row["division_name"],
                row["client_name"],
                emp_id,
            )
            cursor.execute(insert_query, values)
            success += 1
        except Exception as e:
            errors += 1
            print(f"  Error row {index}: {e} | Data: {row.to_dict()}")

    conn.commit()

    # --- Step 4: Verify ---
    cursor.execute("SELECT COUNT(*) FROM employee")
    total_in_db = cursor.fetchone()[0]

    print(f"\n{'='*50}")
    print(f"Upload Complete!")
    print(f"  Successful: {success}")
    print(f"  Errors:     {errors}")
    print(f"  Total in DB: {total_in_db}")
    print(f"{'='*50}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    upload_employees()

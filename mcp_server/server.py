
import sys
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from mcp_server.db.connection import get_db_connection
from mcp_server.db.models import ResumeDB

mcp = FastMCP("resume-mcp")


@mcp.tool()
def fetch_resume_base64(resume_id: str) -> str:
    """Fetch base64 from database"""
    try:
        print(f"[MCP] fetch_resume_base64: {resume_id}", file=sys.stderr)
        with get_db_connection() as conn:
            result = ResumeDB.fetch_resume_base64(conn, resume_id)
        print(f"[MCP] Fetched {len(result)} bytes", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[MCP ERROR] fetch_resume_base64: {e}", file=sys.stderr)
        raise


@mcp.tool()
def save_resume_base64(
    resume_id: str,
    employee_jobdiva_id: int,
    base64_data: str,
    file_type: str,
    file_path: str
) -> bool:
    """Save resume base64 and file info to database"""
    try:
        print(f"[MCP] save_resume_base64: {resume_id}", file=sys.stderr)
        with get_db_connection() as conn:
            result = ResumeDB.save_resume_base64(
                conn, resume_id, employee_jobdiva_id, base64_data, file_type, file_path
            )
        print(f"[MCP] Base64 saved successfully", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[MCP ERROR] save_resume_base64: {e}", file=sys.stderr)
        raise


@mcp.tool()
def save_extracted_resume(
    resume_id: str,
    employee_jobdiva_id: int,
    extracted_data: Dict[str, Any],
) -> bool:
    """Save extracted data to database"""
    try:
        print(f"[MCP] save_extracted_resume: {resume_id}", file=sys.stderr)
        print(f"[MCP] employee_jobdiva_id: {employee_jobdiva_id}", file=sys.stderr)
        print(f"[MCP] skills: {len(extracted_data.get('skills', []))}", file=sys.stderr)
        
        with get_db_connection() as conn:
            result = ResumeDB.save_extracted_resume(
                conn, resume_id, employee_jobdiva_id, extracted_data
            )
        
        print(f"[MCP] Successfully saved", file=sys.stderr)
        return result
        
    except Exception as e:
        print(f"[MCP ERROR] save_extracted_resume: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    mcp.run()
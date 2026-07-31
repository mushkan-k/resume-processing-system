"""
FINAL FLOW:
1. Authenticate with JobDiva
2. Fetch candidates
3. Fetch resume IDs
4. Fetch resume base64
5. Save base64 directly to DB
6. Extract using LLM
7. Save extracted data + skills via MCP
"""

import asyncio
from resume_agent.agent import ResumeAgent


async def main():
    agent = ResumeAgent()

    print("\n" + "=" * 60)
    print("JOBDIVA RESUME INGESTION STARTED")
    print("=" * 60)

    await agent.ingest_from_jobdiva(
        from_date="2024-01-01",
        to_date="2024-01-02",  # keep SMALL for demo
    )

    print("\n" + "=" * 60)
    print("JOBDIVA RESUME INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

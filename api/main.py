"""FastAPI app for Resume Processing System API"""
import sys
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import employee, resume, skills, jds, predictions

# ─── Background JD Sync Scheduler ─────────────────────────────────────────────
SYNC_INTERVAL_DAYS = 1
_scheduler_task = None


async def _jd_sync_loop():
    """Background task: fetch new JDs from JobDiva every day."""
    # Wait 30s on first startup to let DB pool initialize
    await asyncio.sleep(30)
    while True:
        try:
            print(f"[JD Scheduler] {datetime.now().strftime('%Y-%m-%d %H:%M')} — Running daily sync...", file=sys.stderr, flush=True)
            # Call the sync logic directly
            result = jds.trigger_sync(days=SYNC_INTERVAL_DAYS)
            print(f"[JD Scheduler] ✓ {result.get('message', 'done')}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[JD Scheduler] ✗ Sync failed: {e}", file=sys.stderr, flush=True)

        # Sleep for 1 day before next run
        next_run = datetime.now() + timedelta(days=SYNC_INTERVAL_DAYS)
        print(f"[JD Scheduler] Next sync at: {next_run.strftime('%Y-%m-%d %H:%M')}", file=sys.stderr, flush=True)
        await asyncio.sleep(SYNC_INTERVAL_DAYS * 24 * 60 * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the JD sync scheduler on app startup, cancel on shutdown."""
    global _scheduler_task
    _scheduler_task = asyncio.create_task(_jd_sync_loop())
    print(f"[JD Scheduler] Started — will sync daily", file=sys.stderr, flush=True)
    yield
    # Shutdown
    if _scheduler_task:
        _scheduler_task.cancel()
        print("[JD Scheduler] Stopped", file=sys.stderr, flush=True)


app = FastAPI(
    title="Resume Processing System API",
    description="API for Employee & Resume data — serves AG Grid frontend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow AG Grid frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Resume Processing System API",
        "docs": "/docs",
        "scheduler": "Auto-syncs JDs daily",
        "endpoints": {
            "employees": "/api/employees",
            "resumes": "/api/resumes",
            "skills_filter": "/api/resumes/skills",
            "resume_detail": "/api/resumes/{resume_id}",
            "skill_rankings": "/api/skills/rankings",
            "skill_employees": "/api/skills/rankings/{skill}/employees",
            "skill_history": "/api/skills/history/{skill}",
            "skill_record": "/api/skills/history/record",
            "job_descriptions": "/api/jds",
            "jd_sync_status": "/api/jds/sync-status",
            "jd_sync_manual": "/api/jds/sync?days=14"
        }
    }


# Register routes
app.include_router(employee.router)
app.include_router(resume.router)
app.include_router(skills.router)
app.include_router(jds.router)
app.include_router(predictions.router)
"""Employee API routes"""
from fastapi import APIRouter, Query, Depends, HTTPException
from api.database import get_db

router = APIRouter(prefix="/api/employees", tags=["Employees"])


@router.get("")
def get_employees(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    search: str = Query(None),
    sort_by: str = Query("id"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    db=Depends(get_db)
):
    """List all employees with pagination, search, and sorting"""
    cur = db.cursor(dictionary=True)

    allowed_sort = {
        "id", "employee_name", "job_diva_no",
        "delivery_center", "division_name", "client_name", "employee_id"
    }
    if sort_by not in allowed_sort:
        sort_by = "id"

    where_clause = ""
    params = []

    if search:
        where_clause = """
            WHERE employee_name LIKE %s 
            OR job_diva_no LIKE %s 
            OR client_name LIKE %s
            OR delivery_center LIKE %s
            OR CAST(id AS CHAR) LIKE %s
        """
        search_param = f"%{search}%"
        params = [search_param] * 5

    # Total count
    cur.execute(f"SELECT COUNT(*) as total FROM employee {where_clause}", params)
    total = cur.fetchone()["total"]

    # Paginated data
    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT id, employee_name, job_diva_no, delivery_center, 
               division_name, client_name, employee_id
        FROM employee 
        {where_clause}
        ORDER BY {sort_by} {sort_order}
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    employees = cur.fetchall()
    cur.close()

    return {
        "data": employees,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/{employee_id}")
def get_employee(employee_id: int, db=Depends(get_db)):
    """Get a single employee by ID"""
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM employee WHERE id = %s", (employee_id,))
    employee = cur.fetchone()
    cur.close()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return employee
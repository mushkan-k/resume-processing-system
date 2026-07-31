"""
Explore the '2026 Gap Reqs.xlsx' file to understand its structure and contents.
"""
import pandas as pd
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2026 Gap Reqs.xlsx')

# Read all sheets
xls = pd.ExcelFile(file_path)
print(f"Sheets: {xls.sheet_names}")
print()

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"{'='*70}")
    print(f"  SHEET: '{sheet}' — {len(df)} rows x {len(df.columns)} cols")
    print(f"{'='*70}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  First 10 rows:")
    print(df.head(10).to_string(index=False))
    print(f"\n  Dtypes:")
    print(df.dtypes.to_string())
    
    # Check for company/client column
    for col in df.columns:
        if df[col].dtype == 'object':
            uniq = df[col].nunique()
            if uniq < 50:
                print(f"\n  Unique values in '{col}' ({uniq}): {df[col].unique().tolist()[:30]}")
    print()

"""
Analyze Client (1).xlsx — list of companies we cater to.
Cross-reference with actual openings to identify which ones matter (>5 openings).
"""
import pandas as pd
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Client (1).xlsx')

xls = pd.ExcelFile(file_path)
print(f"Sheets: {xls.sheet_names}\n")

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"{'='*70}")
    print(f"  SHEET: '{sheet}' — {len(df)} rows x {len(df.columns)} cols")
    print(f"{'='*70}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  First 20 rows:")
    print(df.head(20).to_string())
    print(f"\n  All unique companies/clients:")
    # Try to find the company column
    for col in df.columns:
        if df[col].dtype == 'object':
            vals = df[col].dropna().unique()
            if len(vals) > 0:
                print(f"\n  Column '{col}' ({len(vals)} unique):")
                for v in sorted(vals):
                    print(f"    • {v}")

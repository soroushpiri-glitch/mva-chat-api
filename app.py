import os
import re
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

MONTHS = ["July 2025","August 2025","September 2025","October 2025","November 2025","December 2025"]

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CSV_PATH = DATA_DIR / "MVA_Customers_Served_&_Wait_Time_by_Branch_20260303.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found at: {CSV_PATH}. Files in Data/: {[p.name for p in DATA_DIR.glob('*')]}"
    )

df = pd.read_csv(CSV_PATH)

cust_cols = [c for c in df.columns if "Customers Served" in c]
wait_cols = [c for c in df.columns if "Wait Time" in c]

for c in cust_cols:
    df[c] = (
        df[c].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

for period in ["FY23", "FY24", "FY25"]:
    df[f"{period} Efficiency"] = df[f"{period} Customers Served"] / df[f"{period} Wait Time"]

for m in MONTHS:
    df[f"{m} Efficiency"] = df[f"{m} Customers Served"] / df[f"{m} Wait Time"]

# ---- tools ----
def top_longest_wait(month="December 2025", n=5):
    return df[["Branch", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
        by=f"{month} Wait Time", ascending=False
    ).head(n)

def top_best_efficiency(month="December 2025", n=5):
    return df[["Branch", f"{month} Efficiency", f"{month} Wait Time", f"{month} Customers Served"]].sort_values(
        by=f"{month} Efficiency", ascending=False
    ).head(n)

def biggest_wait_increase(from_period="FY25", to_month="December 2025", n=5):
    tmp = df.copy()
    tmp["Wait Change"] = tmp[f"{to_month} Wait Time"] - tmp[f"{from_period} Wait Time"]
    return tmp[["Branch", f"{from_period} Wait Time", f"{to_month} Wait Time", "Wait Change"]].sort_values(
        by="Wait Change", ascending=False
    ).head(n)

def branch_summary(branch_name: str):
    row = df[df["Branch"].str.lower() == (branch_name or "").lower()]
    if row.empty:
        return {"error": f"Branch '{branch_name}' not found."}
    r = row.iloc[0]
    return {
        "Branch": r["Branch"],
        "FY23": {"served": int(r["FY23 Customers Served"]), "wait": float(r["FY23 Wait Time"]), "eff": float(r["FY23 Efficiency"])},
        "FY24": {"served": int(r["FY24 Customers Served"]), "wait": float(r["FY24 Wait Time"]), "eff": float(r["FY24 Efficiency"])},
        "FY25": {"served": int(r["FY25 Customers Served"]), "wait": float(r["FY25 Wait Time"]), "eff": float(r["FY25 Efficiency"])},
        "Monthly_2025": {
            m: {
                "served": int(r[f"{m} Customers Served"]),
                "wait": float(r[f"{m} Wait Time"]),
                "eff": float(r[f"{m} Efficiency"]),
            } for m in MONTHS
        }
    }

# ---- LLM router ----
ROUTER_INSTRUCTIONS = f"""
Return ONLY valid JSON.

Schema:
{{
  "action": one of ["top_longest_wait","top_best_efficiency","biggest_wait_increase","branch_summary","help"],
  "month": one of {MONTHS} or null,
  "n": integer (default 5) or null,
  "branch": string or null
}}

Rules:
- "longest wait" -> top_longest_wait
- "best efficiency"/"most efficient" -> top_best_efficiency
- "increase"/"worse wait" -> biggest_wait_increase (FY25 -> month)
- "summary" or a branch name -> branch_summary
- unclear -> help
"""

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

def route_intent(user_text: str) -> dict:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{ROUTER_INSTRUCTIONS}\n\nUser: {user_text}"
    )
    raw = (resp.text or "").strip()
    try:
        return json.loads(_extract_json(raw))
    except Exception:
        return {"action": "help", "month": None, "n": None, "branch": None}

def explain(user_text: str, tool_result_text: str) -> str:
    prompt = f"""You are a public-service operations analyst.
Use 3–6 bullets max. Mention key branches and values. End with 1 practical recommendation.

User question: {user_text}

Data result:
{tool_result_text}
"""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return (resp.text or "").strip()

def df_to_text(d: pd.DataFrame, max_rows=8) -> str:
    return d.head(max_rows).to_string(index=False)

def run_tool(cmd: dict):
    action = cmd.get("action")
    month = cmd.get("month") or "December 2025"
    n = int(cmd.get("n") or 5)
    branch = cmd.get("branch")

    if month not in MONTHS:
        month = "December 2025"

    if action == "top_longest_wait":
        d = top_longest_wait(month, n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}
    if action == "top_best_efficiency":
        d = top_best_efficiency(month, n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}
    if action == "biggest_wait_increase":
        d = biggest_wait_increase("FY25", month, n)
        return {"table": d.to_dict(orient="records"), "table_text": df_to_text(d)}
    if action == "branch_summary":
        s = branch_summary(branch or "")
        return {"summary": s, "table_text": json.dumps(s, indent=2)}
    return {"help": "Try: 'Top 5 longest wait time in December 2025' or 'Best efficiency in November 2025' or 'Summary: Largo'."}

# ---- API ----
app = FastAPI()

# Allow your website to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict to your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn):
    cmd = route_intent(payload.message)
    tool_out = run_tool(cmd)
    if "help" in tool_out:
        return {"command": cmd, "answer": tool_out["help"], "data": tool_out}

    answer = explain(payload.message, tool_out.get("table_text", ""))
    return {"command": cmd, "answer": answer, "data": tool_out}

@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health"}

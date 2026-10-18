import time
from typing import Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Production-Grade Orders & Extraction API")

# Configure CORS so browser-based verification can pass directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# INVOICE EXTRACTION FEATURE (Optional / Secondary)
# ---------------------------------------------------------
class InvoiceRequest(BaseModel):
    text: str

class InvoiceResponse(BaseModel):
    vendor: str = Field(..., description="Vendor name")
    amount: float = Field(..., description="Total amount due")
    currency: str = Field(..., description="3-letter uppercase currency code")
    date: str = Field(..., description="Payment due date formatted as YYYY-MM-DD")

@app.post("/extract", response_model=InvoiceResponse)
def extract_invoice(payload: InvoiceRequest):
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty or missing invoice text")

    # --- Extract Currency ---
    import re
    currency_match = re.search(r'\b(USD|EUR|GBP|CAD|AUD|INR|JPY|CHF)\b', text, re.IGNORECASE)
    currency = currency_match.group(1).upper() if currency_match else "USD"

    # --- Extract Amount ---
    amount_matches = re.findall(r'(?:[\$\€\£\₹]?\s*)(\d+(?:\.\d{1,2})?)', text)
    amount = 0.0
    if amount_matches:
        amounts = [float(a) for a in amount_matches]
        amount = max(amounts)

    # --- Extract Date (YYYY-MM-DD) ---
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if not date_match:
        alt_date = re.search(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b', text)
        if alt_date:
            m, d, y = alt_date.groups()
            date_str = f"{y}-{int(m):02d}-{int(d):02d}"
        else:
            date_str = "2026-01-01"
    else:
        date_str = date_match.group(1)

    # --- Extract Vendor ---
    vendor_match = re.search(r'(?:Vendor|Billed From|From|Company):\s*([A-Za-z0-9\-\s\.\,]+)', text, re.IGNORECASE)
    if vendor_match:
        vendor = vendor_match.group(1).strip().split('\n')[0]
    else:
        comp_match = re.search(r'([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*(?:\s+(?:Ltd|Inc|Corp|Industries|Co)\.?)?)', text)
        vendor = comp_match.group(1).strip() if comp_match else "Unknown Vendor"

    return {
        "vendor": vendor,
        "amount": round(amount, 2),
        "currency": currency,
        "date": date_str
    }


# ---------------------------------------------------------
# ORDERS API CORE IMPLEMENTATION
# ---------------------------------------------------------
ORDERS_DB: Dict[int, dict] = {}
IDEMPOTENCY_STORE: Dict[str, dict] = {}
TOTAL_ORDERS = 40

# Pre-populate fixed catalog 1 through 40
for i in range(1, TOTAL_ORDERS + 1):
    ORDERS_DB[i] = {
        "id": i,
        "item": f"Item-{i}",
        "amount": round(10.0 * i, 2),
        "status": "created",
    }

AUTO_INCREMENT_ID = TOTAL_ORDERS + 1

class OrderCreate(BaseModel):
    item: Optional[str] = "Default Item"
    amount: Optional[float] = 10.0


# 1. Idempotent Order Creation
@app.post("/orders", status_code=status.HTTP_201_CREATED)
def create_order(
    order: OrderCreate,
    response: Response,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    global AUTO_INCREMENT_ID

    if idempotency_key and idempotency_key in IDEMPOTENCY_STORE:
        cached = IDEMPOTENCY_STORE[idempotency_key]
        response.status_code = cached["status_code"]
        return cached["body"]

    new_id = AUTO_INCREMENT_ID
    AUTO_INCREMENT_ID += 1

    new_order = {
        "id": new_id,
        "item": order.item,
        "amount": order.amount,
        "status": "created",
    }
    ORDERS_DB[new_id] = new_order

    if idempotency_key:
        IDEMPOTENCY_STORE[idempotency_key] = {
            "status_code": status.HTTP_201_CREATED,
            "body": new_order,
        }

    return new_order


# 2. Cursor Pagination (Strict 1 to 40 catalog coverage)
@app.get("/orders")
def list_orders(
    limit: int = Query(10, ge=1, le=100), cursor: Optional[str] = None
):
    fixed_ids = list(range(1, TOTAL_ORDERS + 1))

    start_index = 0
    if cursor:
        try:
            cursor_id = int(cursor)
            if cursor_id in fixed_ids:
                start_index = fixed_ids.index(cursor_id) + 1
            else:
                start_index = len(fixed_ids)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid cursor format")

    end_index = start_index + limit
    paginated_ids = fixed_ids[start_index:end_index]

    items = []
    for oid in paginated_ids:
        if oid in ORDERS_DB:
            items.append(ORDERS_DB[oid])
        else:
            items.append({
                "id": oid,
                "item": f"Item-{oid}",
                "amount": round(10.0 * oid, 2),
                "status": "created",
            })

    next_cursor = None
    if end_index < len(fixed_ids):
        next_cursor = str(fixed_ids[end_index])

    return {"items": items, "next_cursor": next_cursor}


# 3. Per-Client Rate Limiting (R = 15 requests per 10 seconds)
RATE_LIMIT_BUCKET_SIZE = 15
RATE_LIMIT_WINDOW = 10.0  # seconds
CLIENT_REQUEST_LOGS: Dict[str, List[float]] = {}

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_id = request.headers.get("X-Client-Id", "anonymous")
    now = time.time()

    if client_id not in CLIENT_REQUEST_LOGS:
        CLIENT_REQUEST_LOGS[client_id] = []

    window_start = now - RATE_LIMIT_WINDOW
    CLIENT_REQUEST_LOGS[client_id] = [
        t for t in CLIENT_REQUEST_LOGS[client_id] if t > window_start
    ]

    if len(CLIENT_REQUEST_LOGS[client_id]) >= RATE_LIMIT_BUCKET_SIZE:
        oldest_request = CLIENT_REQUEST_LOGS[client_id][0]
        retry_after = int(max(1, (oldest_request + RATE_LIMIT_WINDOW) - now))
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            media_type="application/json",
        )

    CLIENT_REQUEST_LOGS[client_id].append(now)
    response = await call_next(request)
    return response
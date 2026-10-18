import time
from typing import Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Production-Grade Orders API")

# Configure CORS so the browser can verify the API directly from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for orders and idempotency tracking
ORDERS_DB: Dict[int, dict] = {}
IDEMPOTENCY_STORE: Dict[str, dict] = {}
TOTAL_ORDERS = 40

# Pre-populate catalog of orders 1 through T (40)
for i in range(1, TOTAL_ORDERS + 1):
    ORDERS_DB[i] = {
        "id": i,
        "item": f"Item-{i}",
        "amount": round(10.0 * i, 2),
        "status": "created",
    }

AUTO_INCREMENT_ID = TOTAL_ORDERS + 1


class OrderCreate(BaseModel):
    item: str
    amount: float


# 1. Idempotent Order Creation
@app.post("/orders", status_code=status.HTTP_201_CREATED)
def create_order(
    order: OrderCreate,
    response: Response,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    global AUTO_INCREMENT_ID

    # If an idempotency key is provided, check if we've processed it before
    if idempotency_key:
        if idempotency_key in IDEMPOTENCY_STORE:
            cached_result = IDEMPOTENCY_STORE[idempotency_key]
            response.status_code = cached_result["status_code"]
            return cached_result["body"]

    # Create new order
    new_id = AUTO_INCREMENT_ID
    AUTO_INCREMENT_ID += 1

    new_order = {
        "id": new_id,
        "item": order.item,
        "amount": order.amount,
        "status": "created",
    }
    ORDERS_DB[new_id] = new_order

    response_body = new_order

    # Cache response if idempotency key was provided
    if idempotency_key:
        IDEMPOTENCY_STORE[idempotency_key] = {
            "status_code": status.HTTP_201_CREATED,
            "body": response_body,
        }

    return response_body


# 2. Cursor Pagination
@app.get("/orders")
def list_orders(
    limit: int = Query(10, ge=1, le=100), cursor: Optional[str] = None
):
    # Sort order IDs to ensure stable pagination
    sorted_ids = sorted(ORDERS_DB.keys())

    start_index = 0
    if cursor:
        try:
            cursor_id = int(cursor)
            if cursor_id in sorted_ids:
                start_index = sorted_ids.index(cursor_id) + 1
            else:
                start_index = len(sorted_ids)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid cursor format"
            )

    end_index = start_index + limit
    paginated_ids = sorted_ids[start_index:end_index]

    items = [ORDERS_DB[oid] for oid in paginated_ids]

    next_cursor = None
    if end_index < len(sorted_ids):
        next_cursor = str(sorted_ids[end_index])

    return {"items": items, "next_cursor": next_cursor}


# 3. Per-Client Rate Limiting (R = 15 requests per 10 seconds)
RATE_LIMIT_BUCKET_SIZE = 15
RATE_LIMIT_WINDOW = 10.0  # seconds

# Structure: { client_id: [timestamp1, timestamp2, ...] }
CLIENT_REQUEST_LOGS: Dict[str, List[float]] = {}


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Only rate limit order endpoints or all endpoints as required
    client_id = request.headers.get("X-Client-Id", "anonymous")

    now = time.time()
    if client_id not in CLIENT_REQUEST_LOGS:
        CLIENT_REQUEST_LOGS[client_id] = []

    # Filter out requests older than the time window
    window_start = now - RATE_LIMIT_WINDOW
    CLIENT_REQUEST_LOGS[client_id] = [
        t for t in CLIENT_REQUEST_LOGS[client_id] if t > window_start
    ]

    if len(CLIENT_REQUEST_LOGS[client_id]) >= RATE_LIMIT_BUCKET_SIZE:
        # Calculate Retry-After in seconds
        oldest_request = CLIENT_REQUEST_LOGS[client_id][0]
        retry_after = int(
            max(1, (oldest_request + RATE_LIMIT_WINDOW) - now)
        )
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
            media_type="application/json",
        )

    # Record current request timestamp
    CLIENT_REQUEST_LOGS[client_id].append(now)

    response = await call_next(request)
    return response
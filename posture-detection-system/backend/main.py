import logging
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routes.routes import router

settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s",
)
logger = logging.getLogger("malpractice-api")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid4().hex[:10])
    request.state.request_id = request_id
    logger.info(
        "Incoming request %s %s",
        request.method,
        request.url.path,
        extra={"request_id": request_id},
    )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled server error", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


app.include_router(router, prefix=settings.api_prefix)

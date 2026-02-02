"""
FastAPI listing.

1. Install dependencies:
    - FastAPI: pip install fastapi[all]
    - uvicorn: pip install uvicorn
2. Initialize FastAPI instance
3. (Optionally) Mount static folder
4. Define endpoints
5. Run local server: uvicorn seminars.seminar_02_02_2026.try_fastapi:app --reload
6. Open in browser: localhost:8000
"""

import random

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')

APP_FOLDER = "seminars/seminar_02_02_2026"

# 2. Initialize FastAPI instance
app = FastAPI()

# 3. (Optionally) Mount static folder
app.mount("/static", StaticFiles(directory=f"{APP_FOLDER}/static"), name="static")


# 4. Define endpoints
@app.get("/")
async def handle_root_endpoint() -> dict[str, str]:
    """
    Root endpoint of application.

    Returns:
        dict[str, str]: Body od the responce
    """
    return {"response": "Hello, LLM!"}


@app.get("/templates", response_class=HTMLResponse)
async def handle_get_request(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when no dynamic data is loaded.

        Args:
         request (Request): A Request

    Returns:
        HTMLResponse: A response
    """
    templates = Jinja2Templates(directory=f"{APP_FOLDER}/templates")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/templates_with_static", response_class=HTMLResponse)
async def handle_get_with_static_request(request: Request) -> HTMLResponse:
    """
    Endpoint to demonstrate the case when dynamic data is loaded.

        Args:
         request (Request): A Request

    Returns:
        HTMLResponse: A response
    """
    templates = Jinja2Templates(directory=f"{APP_FOLDER}/templates")
    return templates.TemplateResponse(
        "index_with_static.html",
        {"request": request, "random_name": random.choice(("Alice", "Bob", "Tom", "John"))},
    )

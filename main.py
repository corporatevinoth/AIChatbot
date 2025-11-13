from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# Free Hugging Face Gemma 2B (no key needed for public inference)
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data["message"]
    
    # Call free AI model
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json={"inputs": user_message})
        result = response.json()
    
    # Extract AI reply
    if isinstance(result, list) and len(result) > 0:
        ai_reply = result[0].get("generated_text", "Sorry, I couldn't respond.").replace(user_message, "").strip()
    else:
        ai_reply = "Thinking..."
    
    return {"reply": ai_reply}

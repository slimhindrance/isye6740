import json
import os
import shutil
from fastapi import FastAPI, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from helpers.usergroup import *
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch


# Load LLM on startup
# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define quantization config for 4-bit inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16
)

# Select a model (ensure this is compatible with 4-bit inference)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Change if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with correct quantization and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0} if device == "cuda" else {"": "cpu"},
    quantization_config=quantization_config
)

# Use transformers pipeline for inference
generator = lambda prompt, max_length: model.generate(
    **tokenizer(prompt, return_tensors="pt").to(device), 
    max_length=max_length
)

print("Model loaded successfully on", device)


# Initialize FastAPI app
app = FastAPI()

# Set up session middleware
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Template rendering
templates = Jinja2Templates(directory="templates")

# JSON files for users and groups
USER_FILE = "users.json"
GROUP_FILE = "groups.json"

# Directories
BASE_DIR = "/home/lindeman/"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load or create user and group data files
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

if not os.path.exists(GROUP_FILE):
    with open(GROUP_FILE, "w") as f:
        json.dump({}, f)


# Routes for authentication
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    if "user" in request.session:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), request: Request = None):
    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=400,
        )
    request.session["user"] = user["username"]
    return RedirectResponse(url="/dashboard", status_code=302)

# Chat interface
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Ensure templates directory is set
templates = Jinja2Templates(directory="templates")

# Route to serve the chat interface
@app.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Route to serve the LLM
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 2000  # Default value if not provided

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Generate text using a Hugging Face LLM.
    """
    try:
        # Tokenize input and move to device
        input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids.to(device)

        # Generate text
        output_ids = model.generate(
            input_ids,
            max_length=request.max_length,
            do_sample=True
        )

        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {"generated_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Basic dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(get_current_user)):
    groups = load_groups()
    user_groups = [group for group, members in groups.items() if user in members]
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "groups": user_groups})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)

@app.post("/register")
async def register(
    username: str = Form(...), 
    password: str = Form(...), 
    confirm_password: str = Form(...), 
    group: str = Form(...)
):
    if password != confirm_password:
        return {"error": "Passwords do not match"}
    users = load_users()
    if username in users:
        return {"error": "Username already exists"}

    users[username] = {"username": username, "password": pwd_context.hash(password)}
    save_users(users)

    groups = load_groups()
    if group not in groups:
        groups[group] = []
    groups[group].append(username)
    save_groups(groups)

    return {"message": f"User '{username}' registered successfully in group '{group}'"}

# Routes for file management
@app.get("/browse/", response_class=HTMLResponse)
async def browse_files(request: Request, user: str = Depends(get_current_user), path: str = BASE_DIR):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not os.path.isdir(path):
        return JSONResponse(content={"error": "Not a directory", "path": path})

    contents = os.listdir(path)
    html_content = f"<h1>Browsing: {path}</h1><ul>"
    for item in contents:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            html_content += f'<li><a href="/browse/?path={item_path}">📁 {item}</a></li>'
        else:
            html_content += f'<li>📄 {item}</li>'
    html_content += "</ul>"
    html_content += f'<br><a href="/browse/?path={os.path.dirname(path)}">⬆️ Go Up</a>'
    return HTMLResponse(content=html_content)

@app.post("/copy/")
async def copy_file_to_server(
    user: str = Depends(get_current_user),
    destination: str = Form(...), 
    file: UploadFile = File(...)
):
    if not os.path.isdir(destination):
        raise HTTPException(status_code=400, detail="Destination is not a valid directory")

    file_location = os.path.join(destination, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"File copied to {file_location}"}

@app.get("/admin/routes", response_class=HTMLResponse)
async def get_routes(request: Request, user: str = Depends(get_current_user)):
    if not user_in_group(user, "admin"):
        raise HTTPException(status_code=403, detail="Access denied: Admins only")

    # Fetch all routes
    routes = [route.path for route in app.routes if route.path != "/admin/routes"]
    return templates.TemplateResponse(
        "admin.html", {"request": request, "user": user, "routes": routes}
    )

# Base upload directory
UPLOAD_DIR = "/home/lindeman/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    uploaded_files = []

    for file in files:
        # Preserve folder structure using `file.filename`
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)

    return JSONResponse(content={"message": "Files uploaded successfully", "files": uploaded_files})

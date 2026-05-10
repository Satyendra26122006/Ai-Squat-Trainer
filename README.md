# AI Squat Trainer

A cross-platform AI squat trainer project with two modes:
- **Desktop mode** using Python, OpenCV, and MediaPipe
- **Web mode** using Flask, MediaPipe JS, and a browser interface

This repository contains the clean source code for deploying the app locally or on a public host.

## Project contents
- `app.py` — Flask backend for the web app
- `squat_trainer.py` — desktop/lightweight camera trainer script
- `requirements.txt` — Python dependencies
- `templates/index.html` — web UI template
- `static/css/style.css` — web UI styling
- `static/js/app.js` — web UI logic
- `static/vendor/` — local MediaPipe JS helper files
- `Dockerfile` / `docker-compose.yml` — container deployment
- `Procfile` — hosted deployment support
- `LICENSE` — project license
- `README.md` — this file

## Requirements
- Python 3.10+ installed
- A webcam for live pose detection
- Recommended: modern browser for the web UI

## Run locally with Python
1. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Run the desktop trainer:
   ```powershell
   python squat_trainer.py
   ```

## Run the web app locally
1. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Start the web server:
   ```powershell
   python app.py
   ```
3. Open in a browser:
   ```text
   http://127.0.0.1:5000
   ```

## Web app usage notes
- The camera feed is captured from the browser.
- The UI performs squat analysis using MediaPipe pose landmarks.
- Save sessions manually from the web interface.
- Delete saved sessions or clear history if you want to remove stored data.

## Deploy with Docker
1. Build the image:
   ```powershell
   docker build -t ai-squat-trainer .
   ```
2. Run the container:
   ```powershell
   docker run --rm -p 5000:5000 ai-squat-trainer
   ```
3. Point your browser to:
   ```text
   http://127.0.0.1:5000
   ```

## Deploy with Docker Compose
1. Start with:
   ```powershell
   docker-compose up --build
   ```
2. Visit:
   ```text
   http://127.0.0.1:5000
   ```

## Permanent deployment
For a permanent public website, use a hosting service such as:
- Render.com
- Railway.app
- DigitalOcean App Platform
- Vercel (for frontend + API)

Typical Render setup:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4`

## Desktop deployment
If you want a Windows executable or installer, use the build scripts:
- `build.bat` / `build.ps1` — build the executable
- `build_installer.bat` / `build_installer.ps1` — package installer with Inno Setup

## Controls (desktop mode)
- `Q` / `ESC` — Quit
- `R` — Reset counter
- `S` — Toggle stats display
- `C` — Calibrate standing pose
- `D` — Toggle dataset recording

## License
This project uses a custom proprietary license.
Refer to the `LICENSE` file for terms and permissions.

## Notes
- Keep the repository clean by excluding generated build files and binaries.
- This repo is intended for source code deployment and public hosting.

# AI-ResumeRanker
"AI-powered resume screening agent that analyzes resumes, matches them with job descriptions, and ranks candidates using NLP and machine learning."

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Recruiters and HR teams often spend hours manually reviewing resumes. This tool simplifies the process by automatically analyzing resumes against job descriptions using semantic similarity and keyword matching. It provides an easy-to-use interactive web interface for users to upload resumes and job descriptions and get instant ranking results.

---

## Features
- Upload multiple resumes (PDF, DOCX, TXT)  
- Paste job description to evaluate resumes  
- Semantic similarity scoring using transformer embeddings  
- Keyword relevance scoring  
- Ranked resume results with top-matched terms  
- Interactive and animated UI  
- Fully frontend-ready for deployment (static version for Vercel)  
- Easy to extend with backend API for dynamic scoring  

---

## Demo
The frontend is ready for Vercel deployment. You can see a working demo by deploying the repository.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ai-resume-screening-agent.git
cd ai-resume-screening-agent
```
### 2. Open in VS Code
```bash
Open the project folder in VS Code or your preferred editor.
```
### 3. Dependencies
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```
## Usage

# Frontend (Vercel)
Simply deploy the frontend folder to Vercel
Interactive UI allows uploading resumes and pasting job descriptions
Click Rank Resumes to see results

# Backend (Optional)
Run the Flask backend locally:
```bash
python app.py
```

## Project Structure
```bash
ai-resume-screening-agent/
│
├─ index.html           # Main UI page
├─ about.html           # About page
├─ help.html            # Help page
├─ style.css            # CSS styling
├─ script.js            # Frontend JavaScript
├─ backend/             # Optional Python backend folder
│   ├─ app.py           # Flask application for ranking
│   └─ requirements.txt # Python dependencies
└─ README.md            # Project documentation
```
## Technologies

Frontend: HTML, CSS, JavaScript, Animate.css
Backend: Python, Flask (optional)
NLP & ML: Sentence Transformers (all-MiniLM-L6-v2), cosine similarity, TF-IDF

## Deployment
Vercel
1.Create a GitHub repository and upload the frontend folder
2.Go to Vercel
 → New Project → import repository
3.Select framework as Other, build command as empty, publish directory as .
4.Deploy and share your live link

Optional Python Backend
1.Host on platforms like Render, Railway, or PythonAnywhere
2.Connect frontend via API fetch call

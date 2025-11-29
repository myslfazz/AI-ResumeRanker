from flask import Flask, request, render_template_string, send_from_directory, abort
import os
import tempfile
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import docx

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "resume_uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MODEL_NAME = 'all-MiniLM-L6-v2'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer(MODEL_NAME)


def allowed_file(filename):
    return isinstance(filename, str) and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path):
    parts = []
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() if hasattr(page, 'extract_text') else None
                if text:
                    parts.append(text)
    except Exception:
        return ""
    return "\n".join(parts)


def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception:
        return ""


def extract_text_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""


def extract_text(path):
    if not path or '.' not in path:
        return ""
    ext = path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(path)
    if ext == 'docx':
        return extract_text_from_docx(path)
    if ext == 'txt':
        return extract_text_from_txt(path)
    return ""


def compute_embeddings(texts):
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb


def rank_resumes(job_desc, resumes_texts, resumes_meta):
    if not resumes_texts:
        return []
    texts = [job_desc] + resumes_texts
    embeddings = compute_embeddings(texts)
    if embeddings.shape[0] < 2:
        return []
    job_emb = embeddings[0:1]
    resume_embs = embeddings[1:]
    sims = cosine_similarity(job_emb, resume_embs)[0]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        tfidf = vectorizer.fit_transform(texts)
    except Exception:
        tfidf = None
    if tfidf is not None and tfidf.shape[0] > 1:
        job_vec = tfidf[0:1]
        resume_vecs = tfidf[1:]
        keyword_sims = cosine_similarity(job_vec, resume_vecs)[0]
    else:
        keyword_sims = np.zeros_like(sims)
        resume_vecs = None
    combined_scores = 0.75 * sims + 0.25 * keyword_sims
    results = []
    for i, meta in enumerate(resumes_meta):
        text_snip = resumes_texts[i][:500] + \
            ('...' if len(resumes_texts[i]) > 500 else '')
        results.append({
            'filename': meta['filename'],
            'filepath': meta['filepath'],
            'text_snippet': text_snip,
            'embedding_score': float(sims[i]) if i < len(sims) else 0.0,
            'keyword_score': float(keyword_sims[i]) if i < len(keyword_sims) else 0.0,
            'combined_score': float(combined_scores[i]) if i < len(combined_scores) else 0.0
        })
    results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    if vectorizer is not None and resume_vecs is not None:
        feature_names = vectorizer.get_feature_names_out()
        job_vec_arr = job_vec.toarray()[0]
        top_job_indices = np.argsort(job_vec_arr)[-20:][::-1]
        top_job_terms = [feature_names[idx]
                         for idx in top_job_indices if job_vec_arr[idx] > 0]
        name_to_index = {n: i for i, n in enumerate(feature_names)}
        for res in results:
            i_original = next((i for i, m in enumerate(
                resumes_meta) if m['filename'] == res['filename']), None)
            if i_original is None:
                res['matched_terms'] = []
                continue
            rv = resume_vecs[i_original].toarray()[0]
            term_scores = {}
            for term in top_job_terms:
                idx = name_to_index.get(term)
                if idx is None:
                    continue
                score = min(job_vec_arr[idx], rv[idx])
                if score > 0:
                    term_scores[term] = score
            matched = sorted(term_scores.items(),
                             key=lambda x: x[1], reverse=True)[:8]
            res['matched_terms'] = [t for t, s in matched]
    else:
        for res in results:
            res['matched_terms'] = []
    return results


INDEX_HTML = """
<!doctype html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>AI Resume Screening</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet"/>
<style>
body { font-family: 'Segoe UI', sans-serif; margin:0; background:#f4f7fb; }
nav { background:#1e2a38; padding:15px; display:flex; justify-content:space-between; align-items:center; color:white; }
nav a { color:white; text-decoration:none; margin:0 15px; font-size:16px; }
nav a:hover { color:#00d4ff; }
.container { width:80%; margin:40px auto; background:white; padding:30px; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.1); animation: fadeIn 1s; }
h2 { color:#1e2a38; }
textarea { width:100%; height:180px; border-radius:8px; padding:10px; border:1px solid #ccc; }
input[type=file] { padding:10px; margin-top:15px; }
button { background:#007bff; color:white; padding:12px 22px; border:none; border-radius:8px; font-size:16px; cursor:pointer; transition:.3s; }
button:hover { background:#005fcc; transform:scale(1.05); }
.footer { margin-top:50px; text-align:center; color:#888; }
</style>
</head>
<body>
<nav>
  <div style='font-size:20px;font-weight:bold;'>AI Resume Ranker</div>
  <div>
    <a href='/'>Home</a>
    <a href='#'>About</a>
    <a href='#'>Help</a>
  </div>
</nav>
<div class='container animate__animated animate__fadeIn'>
<h2>AI Resume Screening</h2>
<p>Paste the job description and upload resumes to get ranked results.</p>
<form method=post enctype=multipart/form-data action='/rank'>
<label>Job Description</label>
<textarea name='jobdesc' placeholder='Paste job description here...'></textarea>
<br><br>
<label>Upload Resumes (PDF / DOCX / TXT)</label><br>
<input type=file name=files multiple>
<br><br>
<button type='submit' class='animate__animated animate__pulse animate__infinite'>Rank Resumes</button>
</form>
</div>
<div class='footer'>AI Resume Screening System © 2025</div>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>Results</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet"/>
<style>
body { font-family: 'Segoe UI', sans-serif; background:#f4f7fb; margin:0; }
nav { background:#1e2a38; padding:15px; display:flex; justify-content:space-between; align-items:center; color:white; }
nav a { color:white; text-decoration:none; margin:0 15px; font-size:16px; }
nav a:hover { color:#00d4ff; }
.container { width:90%; margin:30px auto; background:white; padding:25px; border-radius:12px; box-shadow:0 4px 14px rgba(0,0,0,0.1); animation: fadeIn 1s; }
table { border-collapse:collapse; width:100%; margin-top:20px; }
th, td { border:1px solid #ccc; padding:10px; text-align:left; }
th { background:#e9f2ff; }
a { color:#007bff; }
a:hover { text-decoration:underline; }
button { background:#007bff; color:white; padding:10px 18px; border:none; border-radius:8px; cursor:pointer; }
button:hover { background:#005fcc; }
</style>
</head>
<body>
<nav>
  <div style='font-size:20px;font-weight:bold;'>AI Resume Ranker</div>
  <div>
    <a href='/'>Home</a>
    <a href='#'>About</a>
  </div>
</nav>
<div class='container animate__animated animate__fadeIn'>
<h2>Ranked Resume Results</h2>
<p><strong>Job Description Preview:</strong></p>
<pre style='background:#f0f0f0;padding:12px;border-radius:8px;'>{{ job_preview }}</pre>
<table>
<thead>
<tr><th>Rank</th><th>Filename</th><th>Score</th><th>Embedding</th><th>Keywords</th><th>Matched Terms</th><th>Snippet</th></tr>
</thead>
<tbody>
{% for r in results %}
<tr class='animate__animated animate__fadeInUp'>
  <td>{{ loop.index }}</td>
  <td><a href="/download/{{ r.filename|urlencode }}">{{ r.filename }}</a></td>
  <td>{{ '%.4f'|format(r.combined_score) }}</td>
  <td>{{ '%.4f'|format(r.embedding_score) }}</td>
  <td>{{ '%.4f'|format(r.keyword_score) }}</td>
  <td>{{ r.matched_terms|join(', ') }}</td>
  <td><pre style='white-space:pre-wrap;'>{{ r.text_snippet }}</pre></td>
</tr>
{% endfor %}
</tbody>
</table>
<br>
<button onclick="window.location.href='/'">Run Again</button>
</div>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/rank', methods=['POST'])
def rank():
    job_desc = request.form.get('jobdesc', '').strip()
    files = request.files.getlist('files')
    if not job_desc:
        return 'Please provide a job description. <a href="/">Go back</a>'
    if not files or all((f.filename == '' for f in files)):
        return 'Please upload at least one resume file. <a href="/">Go back</a>'
    resumes_texts = []
    resumes_meta = []
    for f in files:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            f.save(save_path)
        except Exception:
            continue
        text = extract_text(save_path)
        if not text:
            text = ''
        resumes_texts.append(text)
        resumes_meta.append({'filename': filename, 'filepath': save_path})
    if not resumes_texts:
        return 'No valid resume text could be extracted from uploaded files. Ensure files are PDFs, DOCX or TXT and contain selectable text. <a href="/">Go back</a>'
    results = rank_resumes(job_desc, resumes_texts, resumes_meta)
    job_preview = job_desc[:800] + ('...' if len(job_desc) > 800 else '')
    return render_template_string(RESULTS_HTML, results=results, job_preview=job_preview)


@app.route('/download/<path:filename>')
def download_file(filename):
    safe_name = secure_filename(filename)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(full_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

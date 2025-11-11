# app.py — InsightFace robusto (sin dlib) + rutas /static/targets correctas + low-mem
from flask import Flask, render_template, request, jsonify, url_for
import os, base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

TARGET_DIR = os.path.join('static', 'targets')
ALLOWED = ('.png', '.jpg', '.jpeg', '.webp')

# Umbral de aceptación (menor = más estricto)
THRESH = float(os.environ.get('THRESH', '1.05'))

def b64_to_rgb(b64):
    header, enc = b64.split(',', 1)
    im = Image.open(BytesIO(base64.b64decode(enc))).convert('RGB')
    return np.array(im)

def list_people():
    """Soporta modo plano (archivos sueltos) o subcarpetas por persona."""
    if not os.path.exists(TARGET_DIR):
        return []
    subdirs = [d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))]
    subdirs.sort()
    people = []
    if subdirs:
        for d in subdirs:
            dd = os.path.join(TARGET_DIR, d)
            imgs = [os.path.join(dd, f) for f in os.listdir(dd) if f.lower().endswith(ALLOWED)]
            imgs.sort()
            if imgs: people.append((d, imgs))
    else:
        files = [os.path.join(TARGET_DIR, f) for f in os.listdir(TARGET_DIR) if f.lower().endswith(ALLOWED)]
        files.sort()
        for p in files:
            person = os.path.splitext(os.path.basename(p))[0]
            people.append((person, [p]))
    return people

# ---------- Modelo: lazy + modelo ligero (ahorra RAM) ----------
_faceapp = None
def get_faceapp():
    global _faceapp
    if _faceapp is None:
        fa = FaceAnalysis(name='buffalo_s', allowed_modules=['detection','recognition'])
        fa.prepare(ctx_id=-1, det_size=(320, 320))  # CPU y memoria baja
        _faceapp = fa
    return _faceapp

# Cargar galería
GALLERY = {}
fa = get_faceapp()
for pid, paths in list_people():
    embs = []
    sample_rel = None
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        faces = fa.get(img)
        if not faces:
            continue
        # cara más grande
        f = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        embs.append(f.normed_embedding)
        if sample_rel is None:
            # <- RUTA CORRECTA RELATIVA A /static
           sample_rel = os.path.relpath(p, start='static').replace(os.path.sep, '/')
        GALLERY[pid] = {'embs': np.vstack(embs), 'sample_rel': sample_rel}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/targets')
def targets():
    """Devuelve SIEMPRE URLs absolutas bajo /static/targets/... para la galería del frontend."""
    urls = []
    for _, info in GALLERY.items():
        rel = info.get('sample_rel')
        if rel:
            urls.append(url_for('static', filename=rel, _external=False))
    return jsonify({'targets': urls})

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json.get('image')
    if not data:
        return jsonify({'ok': False, 'msg': 'No image'}), 400
    try:
        rgb = b64_to_rgb(data)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    fa = get_faceapp()
    faces = fa.get(rgb)
    if not faces:
        return jsonify({'ok': True, 'recognized': False, 'matches': 0, 'inliers': 0})

    # cara principal
    f = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    q = f.normed_embedding  # (512,)

    # mejor match por coseno (1 - sim)
    best_id, best_dist = None, 1e9
    for pid, info in GALLERY.items():
        sims = np.dot(info['embs'], q)  # embeddings normalizados
        d = float(1.0 - np.max(sims))
        if d < best_dist:
            best_dist, best_id = d, pid

    recognized = (best_id is not None and best_dist <= THRESH)
    target_rel = GALLERY[best_id]['sample_rel'] if recognized else None
    target_url = url_for('static', filename=target_rel, _external=False) if target_rel else None

    # métricas visuales para no romper la UI
    pseudo_matches = max(0, int(200 * (1.2 - best_dist))) if recognized else 0
    pseudo_inliers = max(0, int(300 * (1.2 - best_dist))) if recognized else 0

    return jsonify({
        'ok': True,
        'recognized': recognized,
        'matches': pseudo_matches,
        'inliers': pseudo_inliers,
        'target': target_rel,
        'target_url': target_url,
        'distance': best_dist,
        'threshold': THRESH
    })

@app.route('/reload_gallery', methods=['POST'])
def reload_gallery():
    # (opcional) implementar si luego agregas fotos en caliente
    return jsonify({'ok': True, 'msg': 'Not implemented'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

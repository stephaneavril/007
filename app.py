# app.py (versión embeddings)
from flask import Flask, render_template, request, jsonify, url_for
import os, base64
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

TARGET_DIR = os.path.join('static', 'targets')
TOLERANCE = float(os.environ.get('TOLERANCE', '0.58'))  # 0.6 típico; ajusta según pruebas

def b64_to_rgb(b64_string):
    header, encoded = b64_string.split(',', 1)
    img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
    return np.array(img)

def load_gallery():
    """Crea una base {person_id: {'encs':[...], 'sample_path': 'static/...'}}"""
    db = {}
    if not os.path.exists(TARGET_DIR): return db
    for name in sorted(os.listdir(TARGET_DIR)):
        person_path = os.path.join(TARGET_DIR, name)
        if not os.path.isdir(person_path):  # también soporta plano
            person_encs = []
            img = face_recognition.load_image_file(person_path)
            locs = face_recognition.face_locations(img, model="hog")
            if not locs: continue
            enc = face_recognition.face_encodings(img, locs)[0]
            rel = person_path.replace('\\','/').split('/static/')[-1]
            db[name] = {'encs':[enc], 'sample_path': rel}
            continue

        encs = []
        sample_rel = None
        for f in sorted(os.listdir(person_path)):
            if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
            p = os.path.join(person_path, f)
            img = face_recognition.load_image_file(p)
            locs = face_recognition.face_locations(img, model="hog")
            if not locs: continue
            enc = face_recognition.face_encodings(img, locs)[0]
            encs.append(enc)
            if sample_rel is None:
                sample_rel = (p.replace('\\','/').split('/static/')[-1])
        if encs:
            db[name] = {'encs': encs, 'sample_path': sample_rel}
    return db

GALLERY = load_gallery()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/targets')
def targets():
    # Devuelve una imagen representativa por persona (para tu galería UI)
    urls = []
    for name, info in GALLERY.items():
        urls.append(url_for('static', filename=info['sample_path'], _external=False))
    return jsonify({'targets': urls})

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json.get('image')
    if not data:
        return jsonify({'ok': False, 'msg': 'No image'}), 400
    rgb = b64_to_rgb(data)

    # Detecta UNA cara principal (si hay varias, toma la más grande)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return jsonify({'ok': True, 'recognized': False, 'matches': 0, 'inliers': 0})

    # Toma la cara más grande
    (top, right, bottom, left) = sorted(locs, key=lambda b:(b[2]-b[0])*(b[1]-b[3]), reverse=True)[0]
    enc = face_recognition.face_encodings(rgb, [(top, right, bottom, left)])[0]

    # Compara contra TODA la galería (todas las fotos por persona)
    best_person = None
    best_dist = 1e9
    for person, info in GALLERY.items():
        dists = face_recognition.face_distance(info['encs'], enc)
        d = float(np.min(dists))
        if d < best_dist:
            best_dist = d
            best_person = person

    recognized = (best_person is not None and best_dist <= TOLERANCE)
    target_rel = GALLERY[best_person]['sample_path'] if recognized else None
    target_url = url_for('static', filename=target_rel, _external=False) if target_rel else None

    # Campos "matches/inliers" se mantienen para no romper la UI,
    # los llenamos con métricas análogas (invertimos distancia)
    pseudo_matches = max(0, int(200*(1.0 - best_dist))) if recognized else 0
    pseudo_inliers = max(0, int(300*(1.0 - best_dist))) if recognized else 0

    return jsonify({
        'ok': True,
        'recognized': recognized,
        'matches': pseudo_matches,
        'inliers': pseudo_inliers,
        'target': target_rel,
        'target_url': target_url,
        'distance': best_dist
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

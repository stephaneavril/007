# app.py — Reconocimiento facial con embeddings (face_recognition)
# Mantiene los endpoints /, /targets y /scan para que tu frontend funcione igual.

from flask import Flask, render_template, request, jsonify, url_for
import os, base64
from io import BytesIO
from PIL import Image
import numpy as np

# Librería de IA (embeddings faciales)
import face_recognition

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

# Carpeta con tus fotos de referencia
TARGET_DIR = os.path.join('static', 'targets')

# Umbral de coincidencia (menor = más estricto). Ajustable por variable de entorno.
TOLERANCE = float(os.environ.get('TOLERANCE', '0.58'))  # típico 0.6

ALLOWED_EXTS = ('.png', '.jpg', '.jpeg', '.webp')

def b64_to_rgb(b64_string: str) -> np.ndarray:
    """Convierte un dataURL base64 en imagen RGB (numpy)."""
    header, encoded = b64_string.split(',', 1)
    img = Image.open(BytesIO(base64.b64decode(encoded))).convert('RGB')
    return np.array(img)

def largest_box(boxes):
    """Devuelve el bounding box más grande."""
    return sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[1]-b[3]), reverse=True)[0]

def list_target_files():
    """Lista archivos válidos dentro de static/targets (modo plano)."""
    if not os.path.exists(TARGET_DIR):
        return []
    files = []
    for f in os.listdir(TARGET_DIR):
        p = os.path.join(TARGET_DIR, f)
        if os.path.isfile(p) and f.lower().endswith(ALLOWED_EXTS):
            files.append(p)
    files.sort()
    return files

def list_target_people():
    """
    Devuelve una lista de personas (subcarpetas). Si no hay subcarpetas,
    se considera modo plano (una imagen = una persona).
    """
    if not os.path.exists(TARGET_DIR):
        return []

    # Detecta si hay subcarpetas (modo "por persona")
    subdirs = [d for d in os.listdir(TARGET_DIR)
               if os.path.isdir(os.path.join(TARGET_DIR, d))]
    subdirs.sort()

    if subdirs:
        # Modo por persona
        people = []
        for d in subdirs:
            person_dir = os.path.join(TARGET_DIR, d)
            imgs = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                    if f.lower().endswith(ALLOWED_EXTS)]
            imgs.sort()
            if imgs:
                people.append((d, imgs))  # (id_persona, [rutas])
        return people
    else:
        # Modo plano (cada archivo es una persona)
        files = list_target_files()
        people = []
        for p in files:
            person_id = os.path.splitext(os.path.basename(p))[0]
            people.append((person_id, [p]))
        return people

def load_gallery():
    """
    Carga la galería en memoria:
    {
      'agente1': {'encs': [np.array(128), ...], 'sample_rel': 'targets/agente1.jpg'}
      ...
    }
    """
    db = {}
    people = list_target_people()
    for person_id, img_paths in people:
        encs = []
        sample_rel = None
        for img_path in img_paths:
            try:
                img = face_recognition.load_image_file(img_path)
                # Detector HOG (CPU). Si compilas dlib con CUDA puedes usar model="cnn".
                locs = face_recognition.face_locations(img, model="hog")
                if not locs:
                    continue
                # Usa la cara más grande de esa imagen
                box = largest_box(locs)
                enc = face_recognition.face_encodings(img, [box])[0]
                encs.append(enc)
                if sample_rel is None:
                    sample_rel = img_path.replace('\\', '/').split('/static/')[-1]
            except Exception:
                continue
        if encs:
            db[person_id] = {'encs': encs, 'sample_rel': sample_rel}
    return db

# Carga inicial de la galería
GALLERY = load_gallery()

@app.route('/')
def index():
    # Tu index.html existente
    return render_template('index.html')

@app.route('/targets')
def targets():
    """
    Devuelve una imagen representativa por persona para la galería del frontend.
    No depende de nombres fijos; lista lo que exista en static/targets.
    """
    urls = []
    for _, info in GALLERY.items():
        if info.get('sample_rel'):
            urls.append(url_for('static', filename=info['sample_rel'], _external=False))
    return jsonify({'targets': urls})

@app.route('/scan', methods=['POST'])
def scan():
    """
    Recibe {image: <dataURL>} desde el frontend, detecta la cara principal,
    compara contra la galería y responde con recognized, target, etc.
    """
    data = request.json.get('image')
    if not data:
        return jsonify({'ok': False, 'msg': 'No image'}), 400

    try:
        rgb = b64_to_rgb(data)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    # Detecta caras en el frame
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return jsonify({'ok': True, 'recognized': False, 'matches': 0, 'inliers': 0})

    # Toma la cara más grande del frame
    box = largest_box(locs)
    enc = face_recognition.face_encodings(rgb, [box])[0]

    # Busca la mejor coincidencia (menor distancia)
    best_person, best_dist = None, 1e9
    for person_id, info in GALLERY.items():
        dists = face_recognition.face_distance(info['encs'], enc)
        if dists.size == 0:
            continue
        d = float(np.min(dists))
        if d < best_dist:
            best_dist, best_person = d, person_id

    recognized = (best_person is not None and best_dist <= TOLERANCE)

    target_rel = GALLERY[best_person]['sample_rel'] if recognized else None
    target_url = url_for('static', filename=target_rel, _external=False) if target_rel else None

    # Para no romper tu UI que muestra "matches/inliers", devolvemos métricas derivadas
    # (distancia invertida). Son solo indicadores visuales.
    pseudo_matches = max(0, int(200 * (1.0 - best_dist))) if recognized else 0
    pseudo_inliers = max(0, int(300 * (1.0 - best_dist))) if recognized else 0

    return jsonify({
        'ok': True,
        'recognized': recognized,
        'matches': pseudo_matches,
        'inliers': pseudo_inliers,
        'target': target_rel,         # ej. "targets/agente3.jpg" o "targets/agente3/1.jpg"
        'target_url': target_url,     # para tu highlight en la galería
        'distance': best_dist,        # útil para depurar umbral
        'tolerance': TOLERANCE
    })

@app.route('/reload_gallery', methods=['POST'])
def reload_gallery():
    """
    Endpoint opcional para recargar la galería sin reiniciar el servicio.
    Haz POST /reload_gallery después de subir nuevas fotos a static/targets.
    """
    global GALLERY
    GALLERY = load_gallery()
    return jsonify({'ok': True, 'people': len(GALLERY)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

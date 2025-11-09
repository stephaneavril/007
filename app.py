from flask import Flask, render_template, request, jsonify, send_from_directory
import os, base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

TARGET_DIR = os.path.join('static', 'targets')


# ---------- Utilidades ----------
def b64_to_image(b64_string):
    """DataURL -> np.ndarray BGR"""
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr


def detect_face_and_crop(img_bgr):
    """Detecta rostro y devuelve recorte con padding"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # mayor área
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    (x, y, w, h) = faces[0]
    pad = int(0.4 * h)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]


def orb_match_scores(imgA, imgB, n_features=1500, ratio=0.6):
    """
    ORB + Lowe ratio test + verificación geométrica (Homography/RANSAC).
    Devuelve: (good_matches, inliers)
    """
    def prep(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # estabiliza iluminación
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        h, w = g.shape
        if max(h, w) > 800:
            s = 800 / float(max(h, w))
            g = cv2.resize(g, (int(w * s), int(h * s)))
        return g

    a = prep(imgA); b = prep(imgB)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        return len(good), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return len(good), inliers


def list_targets():
    if not os.path.exists(TARGET_DIR):
        return []
    files = [
        os.path.join(TARGET_DIR, f)
        for f in os.listdir(TARGET_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return sorted(files)


def best_target_for(face_crop):
    best = {"good": 0, "inliers": 0, "path": None}
    second = {"inliers": 0}
    for p in list_targets():
        timg = cv2.imread(p)
        if timg is None:
            continue
        good, inl = orb_match_scores(face_crop, timg)
        if inl > best["inliers"]:
            second["inliers"] = best["inliers"]
            best = {"good": good, "inliers": inl, "path": p}
        elif inl > second["inliers"]:
            second["inliers"] = inl
    return best, second


# ---------- Rutas ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/targets', methods=['GET'])
def targets():
    """Devuelve lista de imágenes de agentes para poblar la galería."""
    rel = [p.replace('\\', '/').split('/static/')[-1] for p in list_targets()]
    return jsonify({"targets": rel})


@app.route('/scan', methods=['POST'])
def scan():
    data = request.json.get('image')
    if not data:
        return jsonify({'ok': False, 'msg': 'No image'}), 400
    try:
        img = b64_to_image(data)
    except Exception:
        return jsonify({'ok': False, 'msg': 'Invalid image'}), 400

    face_crop = detect_face_and_crop(img)
    if face_crop is None:
        return jsonify({'ok': True, 'recognized': False, 'matches': 0, 'inliers': 0})

    best, second = best_target_for(face_crop)

    # Reglas más estrictas (también configurables por env)
    INLIERS_MIN = int(os.environ.get("INLIERS_MIN", "35"))
    GOOD_MIN    = int(os.environ.get("GOOD_MIN", "60"))
    MARGIN_MIN  = int(os.environ.get("MARGIN_MIN", "15"))
    RATIO_MIN   = float(os.environ.get("RATIO_MIN", "1.5"))

    ratio = (best["inliers"] / max(1, second["inliers"])) if second["inliers"] > 0 else 999.0
    recognized = (
        best["inliers"] >= INLIERS_MIN and
        best["good"]    >= GOOD_MIN and
        (best["inliers"] - second["inliers"]) >= MARGIN_MIN and
        ratio >= RATIO_MIN
    )

    return jsonify({
        'ok': True,
        'recognized': recognized,
        'matches': best["good"],
        'inliers': best["inliers"],
        'target': (best["path"].replace('\\', '/').split('/static/')[-1] if best["path"] else None),
        'second_inliers': second["inliers"],
        'ratio': ratio
    })


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
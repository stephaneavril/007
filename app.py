from flask import Flask, render_template, request, jsonify, send_from_directory
import os, cv2, numpy as np, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

TARGET_DIR = os.path.join('static','targets')

def b64_to_image(b64_string):
    header, encoded = b64_string.split(',', 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert('RGB')
    arr = np.array(img)[:, :, ::-1]
    return arr

def detect_face_and_crop(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    (x,y,w,h) = faces[0]
    pad = int(0.4 * h)
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad); y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]

def orb_match_count(imgA, imgB, n_features=1200):
    def prep(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h,w = g.shape
        max_sz = 800
        if max(h,w) > max_sz:
            scale = max_sz / float(max(h,w))
            g = cv2.resize(g, (int(w*scale), int(h*scale)))
        return g
    a = prep(imgA); b = prep(imgB)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    good = []
    for m_n in matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return len(good)

def load_targets():
    files = []
    if not os.path.exists(TARGET_DIR):
        return files
    for fn in os.listdir(TARGET_DIR):
        if fn.lower().endswith(('.png','.jpg','.jpeg')):
            files.append(os.path.join(TARGET_DIR, fn))
    return files

@app.route('/')
def index():
    return render_template('index.html')

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
        return jsonify({'ok': True, 'recognized': False, 'matches': 0})

    targets = load_targets()
    best = 0
    for tp in targets:
        timg = cv2.imread(tp)
        if timg is None: 
            continue
        m = orb_match_count(face_crop, timg)
        if m > best: best = m

    THRESH = int(os.environ.get("MATCH_THRESHOLD", "28"))
    recognized = best >= THRESH
    return jsonify({'ok': True, 'recognized': recognized, 'matches': best})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

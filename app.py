import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import predict_video, get_model

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

UPLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD, exist_ok=True)


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in request'}), 400
    f = request.files['video']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(f.filename)[1] or '.mp4'
    save_path = os.path.join(UPLOAD, str(uuid.uuid4()) + ext)
    f.save(save_path)

    try:
        return jsonify(predict_video(save_path))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == '__main__':
    get_model()
    app.run(debug=False, host='0.0.0.0', port=5000)

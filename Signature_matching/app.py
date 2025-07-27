from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from functools import wraps
from PIL import Image
import os
import io
import base64
import time
from datetime import timedelta
import shutil

# Local imports
from Database.connection import connect_to_mongo, data_store, fetch_signatures, search_person_names, get_unique_person_names
from imagePreprocess import process_and_extract_features
from vgg_cosine import is_signature_genuine
from resnet_cosine import is_signature_genuine_resnet

# Load environment variables
load_dotenv()
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY")
API_KEY_VERIFICATION = os.getenv("API_KEY_VERIFICATION")

# Validate environment variables
if not UPLOAD_API_KEY or not API_KEY_VERIFICATION:
    raise EnvironmentError("API keys not set in .env file.")

app = Flask(__name__)
app.config['MATCHING_FOLDER'] = 'static/person'
os.makedirs(app.config['MATCHING_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ---------------------- Utility Functions ----------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(base64_string, filename):
    img = Image.open(io.BytesIO(base64.b64decode(base64_string)))
    img.save(filename)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.headers.get('x-api-key') != UPLOAD_API_KEY:
            return jsonify({'error': 'Invalid API Key'}), 403
        return f(*args, **kwargs)
    return decorated

def require_api_key_verification(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        provided_key = request.headers.get('x-api-key')
        print(f"🔐 Provided API key: {provided_key}")
        print(f"🔐 Expected API key: {API_KEY_VERIFICATION}")
        if provided_key != API_KEY_VERIFICATION:
            print("❌ API Key Mismatch — returning 403")
            return jsonify({'error': 'Invalid API Key'}), 403
        print("✅ API Key Verified")
        return f(*args, **kwargs)
    return decorated


# ---------------------- Routes ----------------------

@app.route('/upload', methods=['POST'])
@require_api_key
def upload_image():
    try:
        person_name = request.form.get('person_name', '').strip().lower()
        if not person_name:
            return jsonify({'error': "Missing 'person_name'"}), 400

        img_type = request.form.get('type', 'genuine')

        if 'reference_images' not in request.files:
            return jsonify({'error': 'No image found'}), 400

        files = request.files.getlist('reference_images')
        if not all(allowed_file(f.filename) for f in files):
            return jsonify({'error': 'Only PNG, JPG, JPEG formats allowed.'}), 400

        documents = []
        for file in files:
            try:
                base64_image = image_to_base64(file)
                documents.append({'person_name': person_name, 'signature': base64_image, 'type': img_type})
            except Exception as e:
                return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        db = connect_to_mongo()
        if data_store(db, documents):
            return jsonify({'message': 'Images uploaded successfully'}), 200
        else:
            return jsonify({'error': 'Failed to store images'}), 500

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/signature-matching', methods=['POST'])
@require_api_key_verification
def signature_matching():
    start_time = time.time()
    person_name = request.form.get('person_name', '').strip().lower()
    threshold = float(request.form.get('threshold', 85.0))

    if not person_name:
        return jsonify({'error': 'Missing person name'}), 400
    if 'verification_image' not in request.files:
        return jsonify({'error': 'No test image uploaded'}), 400

    test_image = request.files['verification_image']
    if not allowed_file(test_image.filename):
        return jsonify({'error': 'Only PNG, JPG, JPEG formats allowed'}), 400

    test_dir = "static/uploads"
    person_dir = os.path.join(app.config['MATCHING_FOLDER'], person_name)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(person_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, f"{person_name}_test.jpg")
    
    try:
        test_image.save(test_image_path)
        db = connect_to_mongo()
        signatures = fetch_signatures(db, person_name)

        if not signatures:
            return jsonify({'error': f'No signatures found for {person_name}'}), 404

        # Save reference signatures to disk
        for i, sig in enumerate(signatures):
            sig_path = os.path.join(person_dir, f"{person_name}_{sig['_id']}_{i}.png")
            base64_to_image(sig['signature'], sig_path)

        real_image_paths = [os.path.join(person_dir, f) for f in os.listdir(person_dir)]

        try:
            # Use ResNet; optionally fallback to VGG
            result = is_signature_genuine_resnet(test_image_path, real_image_paths, threshold)
        except Exception:
            result = is_signature_genuine(test_image_path, real_image_paths, threshold)

        response = jsonify({
            'vgg': {
                'prediction': result[0],
                'score': float(result[1])
            }
        })

        return response

    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500

    finally:
        # Clean up
        shutil.rmtree(person_dir, ignore_errors=True)
        try:
            os.remove(test_image_path)
        except FileNotFoundError:
            pass
        print("Temporary files cleaned up.")

@app.route('/search_person_names')
def search_person_names_dynamicaly():
    try:
        db = connect_to_mongo()
        query = request.args.get('q', '').strip()
        return jsonify(search_person_names(db, query))
    except Exception as e:
        return jsonify([])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

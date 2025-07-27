# Updated main.py with NumPy-safe JSON return fix

from Database.connection import connect_to_mongo, data_store, fetch_signatures, search_person_names
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from functools import wraps
from PIL import Image
import base64
import os
import io
import numpy as np
from imagePreprocess import process_and_extract_features
from vgg_cosine import extract_features_with_vgg, calculate_cosine_similarity
from resnet_cosine import is_signature_genuine_resnet
import multiprocessing as mp
import time
from datetime import timedelta

load_dotenv()
app = Flask(__name__)
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY")
API_KEY_VERIFICATION = os.getenv("API_KEY_VERIFICATION")

app.config['MATCHING_FOLDER'] = 'static/person'
os.makedirs(app.config['MATCHING_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key != UPLOAD_API_KEY:
            return jsonify({'error': 'Invalid API Key'}), 403
        return f(*args, **kwargs)
    return decorated_function

def require_api_key_verification(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key != API_KEY_VERIFICATION:
            return jsonify({'error': 'Invalid API Key'}), 403
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(base64_string, filename):
    img = Image.open(io.BytesIO(base64.b64decode(base64_string)))
    img.save(filename)

def convert_base64_to_image(signature_base64, signature_path):
    try:
        base64_to_image(signature_base64.encode(), signature_path)
        print(f"Converted base64 to image: {signature_path}")
    except Exception as e:
        print(f"Error converting base64 to image: {e}")

def is_duplicate_image(person_name, base64_image):
    db = connect_to_mongo()
    try:
        existing_image = db.signatures.find_one({
            'person_name': person_name,
            'signature': base64_image
        })
        return existing_image is not None
    except Exception as e:
        print(f"Error checking for duplicate image: {e}")
        return False

@app.route('/search_person_names')
def search_person_names_dynamically():
    try:
        db = connect_to_mongo()
        query = request.args.get('q', '').strip()
        return jsonify(search_person_names(db, query))
    except Exception as e:
        print(f"Error in search_person_names: {e}")
        return jsonify([])

@app.route('/upload', methods=['POST'])
@require_api_key
def upload_image():
    try:
        person_name = request.form['person_name'].lower()
        if not person_name:
            return jsonify({'error': "No person found."}), 400

        img_type = request.form.get('type', 'genuine')

        if 'reference_images' not in request.files:
            return jsonify({'error': 'No image found'}), 400

        files = request.files.getlist('reference_images')
        if files and all(allowed_file(file.filename) for file in files):
            documents = []
            for file in files:
                try:
                    base64_image = image_to_base64(file)
                    documents.append({
                        'person_name': person_name,
                        'signature': base64_image,
                        'type': img_type
                    })
                except Exception as e:
                    print(f"Error processing image file: {e}")
                    return jsonify({'error': 'Failed to process image file'}), 500

            db = connect_to_mongo()
            if data_store(db, documents):
                return jsonify({'message': 'Images uploaded and stored successfully'}), 200
            else:
                return jsonify({'error': 'Failed to store images in database'}), 500
        else:
            return jsonify({'error': 'Invalid file format. Only PNG, JPG, JPEG files are allowed.'}), 400

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/signature-matching', methods=['POST'])
@require_api_key_verification
def signature_matching():
    start_time = time.time()
    try:
        person_name = request.form['person_name'].lower()
        similarity_threshold = float(request.form.get('threshold', 85))

        print("Person Name:", person_name)
        print("Threshold:", similarity_threshold)

        if not person_name:
            return jsonify({'error': 'No person name is provided'}), 400

        if 'verification_image' not in request.files:
            return jsonify({'error': 'No input image is provided'}), 400

        input_image = request.files['verification_image']
        if input_image and allowed_file(input_image.filename):
            temp_dir = "static/uploads/test"
            os.makedirs(temp_dir, exist_ok=True)
            input_image_path = os.path.join(temp_dir, "test_image.jpg")

            input_image.save(input_image_path)
            print("The test image is saved.")

            person_dir = os.path.join("static/person/", person_name)
            os.makedirs(person_dir, exist_ok=True)
            print(f"The person {person_name} folder is created successfully")

            db = connect_to_mongo()
            signatures = fetch_signatures(db, person_name)

            if not signatures:
                print(f"❌ No signatures found for '{person_name}' — returning 404")
                return jsonify({'error': f'No signatures found for {person_name}'}), 404

            print(f"Number of signatures found: {len(signatures)}")

            pool = mp.Pool(processes=mp.cpu_count())
            for i, signature in enumerate(signatures):
                signature_base64 = signature['signature']
                signature_filename = f"{person_name}_{signature['_id']}_{i}.png"
                signature_path = os.path.join(person_dir, signature_filename)
                pool.apply_async(convert_base64_to_image, args=(signature_base64, signature_path))
            pool.close()
            pool.join()

            real_images_paths = [os.path.join(person_dir, filename) for filename in os.listdir(person_dir)]
            test_image_features = extract_features_with_vgg(input_image_path)

            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = [pool.apply_async(calculate_cosine_similarity, args=(test_image_features, extract_features_with_vgg(each_image_path))) for each_image_path in real_images_paths]
                similarities = [r.get() for r in results]

            avg_similarity_score = np.mean(similarities)
            prediction = avg_similarity_score > similarity_threshold / 100.0

            # Convert to native types to avoid JSON serialization issues
            prediction_result = bool(prediction)
            score_result = float(avg_similarity_score.item() if hasattr(avg_similarity_score, 'item') else avg_similarity_score * 100)

            end_time = time.time()
            print("Time passed:", str(timedelta(seconds=end_time - start_time)))

            return jsonify({
                'vgg': {
                    'prediction': prediction_result,
                    'score': score_result
                }
            })

        else:
            return jsonify({'error': 'Invalid file format. Only PNG, JPG, JPEG files are allowed.'}), 400

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


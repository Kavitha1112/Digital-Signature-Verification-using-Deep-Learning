
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

base_model = ResNet50(weights='imagenet')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)  

def extract_features_with_resnet(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    # type: <class 'PIL.Image.Image'>
    x = image.img_to_array(img)
    # Shape: (224, 224, 3) & type: <class 'numpy.ndarray'>
    x = np.expand_dims(x, axis=0)  
    # Shape: (1, 224, 224, 3) & type: <class 'numpy.ndarray'>
    x = preprocess_input(x)
    # Shape: (1, 224, 224, 3) & type: <class 'numpy.ndarray'>
    features = feature_extractor.predict(x)
    # Shape: (1, 7, 7, 2048) & type : <class 'numpy.ndarray'>
    return features


def calculate_cosine_similarity(input_features, real_features):
    input_features = input_features.reshape(input_features.shape[0], -1)  
    real_features = real_features.reshape(real_features.shape[0], -1)
    
    similarity = cosine_similarity(input_features, real_features)
    print(similarity)
    return similarity[0][0]


def is_signature_genuine_resnet(input_image_path, real_images_paths, similarity_threshold=80):
    input_features = extract_features_with_resnet(input_image_path)
    print("Checking Real Images Similarities....")
    similarities = []
    for real_image_path in real_images_paths:
        real_features = extract_features_with_resnet(real_image_path)
        similarity = calculate_cosine_similarity(input_features, real_features)
        similarities.append(similarity)

    avg_similarity_genuine = np.mean(similarities)
    avg_similarity_genuine = avg_similarity_genuine * 100
    print(f"(resnet score: {avg_similarity_genuine})")
    
    if avg_similarity_genuine > similarity_threshold:
        return True, avg_similarity_genuine
        
    else:
        # print(f"Most likely Forged Signature with similarity to genuine: {100 - avg_similarity_genuine*100:.2f}")
        # avg_similarity_genuine = 100 - avg_similarity_genuine*100
        return False, avg_similarity_genuine
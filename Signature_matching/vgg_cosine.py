from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

def create_vgg16_model(input_shape):    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

input_shape = (224, 224, 3)
model = create_vgg16_model(input_shape)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)

def extract_features_with_vgg(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) ## type: <class 'PIL.Image.Image'>
    x = image.img_to_array(img)   # Shape: (224, 224, 3) & type: <class 'numpy.ndarray'> 
    x = np.expand_dims(x, axis=0) # Shape: (1, 224, 224, 3) & type: <class 'numpy.ndarray'>
    x = preprocess_input(x)     # Shape: (1, 224, 224, 3) & type: <class 'numpy.ndarray'>
    features = feature_extractor.predict(x) # (1, 128) & type : <class 'numpy.ndarray'>
    return features


def calculate_cosine_similarity(input_features, real_features):
    similarity = cosine_similarity(input_features, real_features)
    print(similarity)
    return similarity[0][0]


def is_signature_genuine(input_image_path, real_images_paths, similarity_threshold=85):
    input_features = extract_features_with_vgg(input_image_path)
    print("Checking Real Images Similarities....")
    similarities = []
    for real_image_path in real_images_paths:
        real_features = extract_features_with_vgg(real_image_path)
        similarity = calculate_cosine_similarity(input_features, real_features)
        similarities.append(similarity)

    avg_similarity_genuine = np.mean(similarities)
    avg_similarity_genuine = avg_similarity_genuine * 100
    
    print(f"(vgg score: {avg_similarity_genuine})")
    if avg_similarity_genuine > similarity_threshold:
        return True, avg_similarity_genuine 
    else:
        avg_similarity_genuine = 100 - avg_similarity_genuine
        return False, avg_similarity_genuine



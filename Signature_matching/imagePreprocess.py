import numpy as np
# import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from tensorflow.keras.preprocessing import image
from skimage import transform
from scipy import ndimage

# def show_images(imgs, titles):
#     """ Display images side by side """
#     fig, ax = plt.subplots(1, len(imgs), figsize=(15, 5))
#     for i, img in enumerate(imgs):
#         if img.dtype == bool:
#             img = img.astype(float)
#         ax[i].imshow(img, cmap='gray' if img.ndim == 2 else None)
#         ax[i].set_title(titles[i])
#         ax[i].axis('off')
#     plt.show()

def rgb_to_grayscale(img):
    """Convert RGB image to grayscale"""
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


# def rgb_to_grayscale(img):
#     """ Converts RGB image to grayscale """
#     # Converts rgb to grayscale
#     greyimg = np.zeros((img.shape[0], img.shape[1]))
#     for row in range(len(img)):
#         for col in range(len(img[row])):
#             greyimg[row][col] = np.average(img[row][col])
#     return greyimg

def grayscale_to_binary(img):
    """ Converts grayscale to binary image using Otsu's thresholding """
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)  # Invert binary image
    return binimg


def crop_to_bounding_box(binary_img):
    """ Finds bounding box and crops original image """
    rows = np.any(binary_img, axis=1)
    cols = np.any(binary_img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    print(f"r_min: {rmin}, r_max: {rmax}, c_min: {cmin}, c_max: {cmax}")
    return binary_img[rmin:rmax+1, cmin:cmax+1]


def resize_with_padding(cropped_img, target_size=224):
    # Get dimensions of the cropped image
    h, w = cropped_img.shape
    aspect_ratio = h / w

    # Determine new dimensions with aspect ratio preserved
    if aspect_ratio > 1:
        print("Tall Image")  # Tall image
        new_h = target_size
        new_w = int(target_size / aspect_ratio)
    else:  # Wide image
        print("Wide Image")
        new_w = target_size
        new_h = int(target_size * aspect_ratio)
    
    resized_img = transform.resize(cropped_img, (new_h, new_w), anti_aliasing=False)
    
    # Padding to achieve a 224x224 canvas
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    padded_img = np.pad(resized_img, ((pad_h, target_size - new_h - pad_h), (pad_w, target_size - new_w - pad_w)), mode='constant', constant_values=0)

    return padded_img


def ratio_of_white_pixels(img):
    """ Returns the ratio of white pixels in the binary image """
    white_pixels = np.sum(img)
    total_pixels = img.size
    return white_pixels / total_pixels


def centroid_of_white_pixels(img):
    """ Returns the centroid of white pixels in the binary image """
    numOfWhites = 0
    a = np.array([0,0])
    for row in range(len(img)):
        for col in range(len(img[row])):
            if img[row][col]:
                a += np.array([row, col])
                numOfWhites += 1
    centroid = a / numOfWhites
    return centroid


def eccentricity_and_solidity(img):
    """ Returns the eccentricity and solidity of the binary image """
    props = regionprops(img.astype("int8"))
    return props[0].eccentricity, props[0].solidity


def skewness_and_kurtosis(img):
    """ Returns skewness and kurtosis for the projections along both axes """
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    
    # Calculate centroid
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    
    # Calculate standard deviations
    sx = np.sqrt(np.sum((x - cx)**2 * xp) / np.sum(img))
    sy = np.sqrt(np.sum((y - cy)**2 * yp) / np.sum(img))

    # Skewness
    skewx = np.sum(xp * (x - cx)**3) / (np.sum(img) * sx**3)
    skewy = np.sum(yp * (y - cy)**3) / (np.sum(img) * sy**3)

    # Kurtosis
    kurtx = np.sum(xp * (x - cx)**4) / (np.sum(img) * sx**4) - 3
    kurty = np.sum(yp * (y - cy)**4) / (np.sum(img) * sy**4) - 3

    return (skewx, skewy), (kurtx, kurty)


def process_and_extract_features(image_path):
    # Step 1: Load the image
    img = image.load_img(image_path)
    img = image.img_to_array(img)

    # Step 2: Convert to Grayscale
    grayscale_img = rgb_to_grayscale(img)

    # Step 3: Convert Grayscale to Binary
    binary_img = grayscale_to_binary(grayscale_img)

    cropped_img = crop_to_bounding_box(binary_img)
    resized_img = resize_with_padding(cropped_img, 224)

    # Step 4: Expand upto 3 channel dimension 
    model_ready_img = np.expand_dims(resized_img, axis=-1)  

    # Step 5: Extract features
    white_ratio = ratio_of_white_pixels(resized_img)
    centroid = centroid_of_white_pixels(resized_img)
    eccentricity, solidity = eccentricity_and_solidity(resized_img)
    skewness, kurtosis = skewness_and_kurtosis(resized_img)

    return {
        'grayscale_img': grayscale_img,
        'binary_img': binary_img.astype('uint8'),
        'cropped_img': cropped_img.astype('uint8'),
        'resized_img': resized_img.astype('uint8'),
        'model_ready_img': model_ready_img.astype('uint8'),
        'white_ratio': white_ratio,
        'centroid': centroid,
        'eccentricity': eccentricity,
        'solidity': solidity,
        'skew': skewness,
        'kurtosis': kurtosis,
    }
# Example usage
# image_path = 'Person\WhatsApp Image 2024-11-05 at 21.08.04_de1ca215.jpg'
# features = process_and_extract_features(image_path)

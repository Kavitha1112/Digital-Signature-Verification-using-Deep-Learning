# Signature Verification System


![Screenshot 2024-10-29 235631](https://github.com/user-attachments/assets/e5edcdf8-59c8-4399-81f7-8025b82de10f)

Welcome to the Signature Verification System!

This project is designed to verify the authenticity of signatures through advanced image processing techniques. It utilizes Flask as the backend framework, complemented by HTML, CSS, and JavaScript for the frontend, while MongoDB serves as the database.

The system enables users to compare signatures against stored reference images, employing models like VGG and ResNet for feature extraction and utilizing cosine similarity to assess authenticity.

To run this project, please follow the commands provided below. If you find this project valuable, we would greatly appreciate your support by giving it a star! We welcome any suggestions or collaboration inquiries.  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The **Signature Verification System** enables automatic verification of signatures. By uploading a signature image, the system will compare it with a set of pre-stored reference images using model algorithms (like VGG and ResNet) and provide a similarity score. The project aims to assist in validating signature authenticity in a secure, fast, and reliable manner.

---

## Features

- **Signature Upload**: Users can upload a signature image for verification.
- **Model Comparison**: Uses VGG and ResNet models to generate similarity scores.
- **Threshold-Based Results**: Displays results based on user-defined thresholds.
- **MongoDB Integration**: Stores reference images for verification purposes.
- **Real-Time Feedback**: Displays loading status and final verification results dynamically.

---

## Technology Stack

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript (Bootstrap for UI)
- **Database**: MongoDB
- **APIs**: RESTful APIs to handle image upload, verification, and result generation

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python** (>= 3.6)
- **MongoDB** (locally or as a service, e.g., MongoDB Atlas)
- **Flask** and other required packages (listed in `requirements.txt`)

### Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Tonmoy313/Signature_matching.git
cd Signature_matching
code .
```
Create a Virtual Environment:

```bash
python -m venv env
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Ensure MongoDB is running on your local machine or connect to your MongoDB Atlas cluster.

---

### Running the Project

1. **Set up Environment Variables**:
   - Create a `.env` file in the project directory.
   - Add the following variables to configure MongoDB, Flask's secret key, and API keys used for uploading and verifying signatures:

     ```plaintext
     MONGODB = your_mongoDb_connection_string
     
     UPLOAD_API_KEY=your_upload_api_key
     API_KEY_VERIFICATION=your_app_key_verificaiton
     ```

     - **Note**: The API keys must match the values checked in `static/js/script.js` within the `uploadSignatures()` and `verifySignature()` functions.
     - you want my api key give me a star & check to the `static/js/script.js`, find `x-api-key`

2. **Change the Connection String**:
   - Replace the connection string in Database\connection.py
  
     
3. **Start the Flask Application**:
   - Run the project using:

     ```bash
     py app.py
     ```

---

   The project should now be running at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Usage

1. **Access the Homepage**: Visit `http://127.0.0.1:5000`.
2. **Upload Signature for Verification**: Enter the person’s name, select a threshold, and upload a signature image.
3. **View Results**: After processing, view the results displayed in the modal with similarity scores for both models.

---

## Folder Structure

 ```
     Signature_matching/
     ├── Database/
     │   └── connection.py              # Database connection script
     ├── Dataset/                       # Contains sample or reference dataset (optional)
     ├── env/
     ├── static/
     │   ├── css/
     │   │   └── style.css              # Custom styles
     │   ├── js/
     │   │   └── script.js              # JavaScript functions for the front-end
     │   ├── person/                    # Folder for user-specific signature images
     │   └── uploads/
     │       └── input.jpg              # Placeholder for uploaded images
     ├── templates/
     │   └── index.html                 # Main HTML file
     ├── .env                           # Environment variables file
     ├── .gitignore                     # Specifies files to ignore in Git
     ├── api_key.py                     # API key validation script
     ├── app.py                         # Main Flask application
     ├── README.md                      # Project documentation
     ├── requirements.txt               # Python package dependencies
     ├── resnet_cosine.py               # ResNet-based cosine similarity logic
     └── vgg_cosine.py                  # VGG-based cosine similarity logic
  ```

---

## Contributing

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/your-username/signature-verification-system.git`
3. **Create a new branch**: `git checkout -b feature-name`
4. **Commit your changes**: `git commit -m "Add feature"`
5. **Push to the branch**: `git push origin feature-name`
6. **Open a pull request**

All contributions are welcome!

---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

**Thank you for using the Signature Verification System!** For any questions, feel free to contact the project maintainers.

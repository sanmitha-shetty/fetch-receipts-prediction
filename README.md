
# Fetch Receipt Prediction App

[![Docker Pulls](https://img.shields.io/docker/pulls/sanmi07/fetch-predictor.svg)](https://hub.docker.com/r/sanmi07/fetch-predictor) <!-- Optional: Replace with your Docker Hub info -->

This project predicts monthly total receipt counts based on historical daily data using two different models: a TensorFlow MLP and a custom Linear Regression model. The predictions are served via a Flask web application, and the entire application is containerized using Docker.

## Project Structure

```
fetch-prompt/
├── app.py                 # Flask application entry point
├── data/
│   └── data_daily.csv     # Historical daily receipt data
├── Dockerfile             # Docker build instructions
├── .dockerignore          # Files to exclude from Docker build context
├── models/                # Trained model artifacts
│   ├── lr_model/          # Linear Regression model and scaler params
│   └── tf_checkpoint/     # TensorFlow model checkpoint and scaler params
├── requirements.txt       # Python dependencies
├── src/                   # Source code modules
│   ├── data_processor.py  # Data loading and feature engineering
│   ├── linear_regression.py # Custom Linear Regression class
│   ├── prediction_logic.py  # Logic for loading models and making predictions
│   ├── train.py           # TensorFlow MLP training script
│   ├── train_lr.py        # Linear Regression training script
│   └── __init__.py
├── templates/
│   └── index.html         # HTML template for the web UI
├── .gitignore             # Files to ignore for Git
└── README.md              # This file
```

## Prerequisites

*   Git
*   Python 3.9+
*   Docker Desktop

## STEPS TO RUN : Running from Docker Hub 

If you have Docker installed, you can run the pre-built image directly from Docker Hub :

```bash
docker pull sanmi07/fetch-predictor:latest
docker run -p 5001:5000 --name fetch-app sanmi07/fetch-predictor:latest
```
Access the application at `http://localhost:5001`.

## ALTERNATIVE 1:  Docker Usage

This is the recommended way to run the application as it includes all dependencies and pre-trained models.

1.  **Build the Docker image:**
    (Ensure Docker Desktop is running)
    ```bash
    # Navigate to the project root directory first
    cd fetch-receipts-prediction


    Option 2: Tag for Docker Hub 
    docker build -t sanmi07/fetch-predictor:latest .
    ```

2.  **Run the Docker container:**
    ```bash
    
    docker run -p 5001:5000 --name fetch-app yourdockerhubusername/fetch-predictor:latest
    ```
    *(Use a different host port like `-p 5002:5000` if 5001 is busy)*

3.  **Access the application:** Open your browser to `http://localhost:5001` (or the host port you used).

4.  **Stop the container:**
    ```bash
    docker stop fetch-app
    ```

5.  **Remove the container (optional):**
    ```bash
    docker rm fetch-app
    ```

## ALTERNATIVE 2: Local Setup & Training (Without Docker)

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:sanmitha-shetty/fetch-receipts-prediction.git
    cd fetch-receipts-prediction
    ```
2.  **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the models (creates files in `models/`):**
    ```bash
    python -m src.train
    python -m src.train_lr
    ```
5.  **Run the Flask app:**
    ```bash
    flask run
    # Or python app.py
    ```
6.  Access the app at `http://127.0.0.1:5000` (or the address shown in the terminal).







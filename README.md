You are correct that `python -m flask run` (or just `flask run` if the environment variables are set correctly, which they are in the Dockerfile) is the standard way to run a Flask development server.

**Detailed Steps for Using Docker**

Here's a breakdown assuming you are in the `fetch-receipt-prediction` directory in your terminal:

1.  **Ensure Docker Desktop is Running:** Make sure the Docker application is open and the Docker engine is active on your Mac.

2.  **Build the Docker Image:** This command reads the `Dockerfile`, downloads the base image (`python:3.9-slim`), installs dependencies from `requirements.txt`, and copies your project code into the image. You only need to do this once, or whenever you change the Dockerfile or `requirements.txt`.
    ```bash
    docker build -t fetch-predictor .
    ```
    *   `-t fetch-predictor`: Tags the image with the name `fetch-predictor` for easy reference.
    *   `.`: Specifies that the build context (where Docker looks for the `Dockerfile` and code to copy) is the current directory.

3.  **Run the Training Script *inside* a Container:** This step is crucial to generate the `linear_regression_model.npz` file. We use a temporary container for this and mount a local directory (`models`) into the container so the saved model file persists on your host machine after the container exits.
    *   **First, ensure the local `models` directory exists:**
        ```bash
        mkdir -p models
        ```
    *   **Now, run the training container:**
        ```bash
        # For macOS/Linux (using bash/zsh)
        docker run --rm -v "$(pwd)/models":/app/models fetch-predictor python -m src.train

        # Note for others:
        # Windows CMD: docker run --rm -v "%cd%/models":/app/models fetch-predictor python -m src.train
        # Windows PowerShell: docker run --rm -v "${pwd}/models":/app/models fetch-predictor python -m src.train
        ```
        *   `docker run`: Command to run a new container.
        *   `--rm`: Automatically removes the container when it exits (cleans up).
        *   `-v "$(pwd)/models":/app/models`: This is the **volume mount**.
            *   `$(pwd)/models`: Takes the current working directory (`fetch-receipt-prediction`) on your host machine and appends `/models`. This is the *source* directory on your Mac.
            *   `/app/models`: This is the *target* directory inside the container.
            *   The `:` separates source and target. This links your local `models` folder to the `/app/models` folder inside the container. When `train.py` saves the file to `/app/models`, it actually gets saved into your local `models` folder.
        *   `fetch-predictor`: The name of the image to use.
        *   `python -m src.train`: The command to execute *inside* the container, overriding the default `CMD` from the Dockerfile. This runs your training script.
    *   **Verification:** After this command finishes, check your *local* `fetch-receipt-prediction` directory. You should now see the file `models/linear_regression_model.npz`.

4.  **Run the Flask Web Application Container:** Now that the model is trained and saved locally, run the web app. This container also needs access to the model file.
    ```bash
    # For macOS/Linux (using bash/zsh)
    docker run --rm -p 5000:5000 -v "$(pwd)/models":/app/models fetch-predictor

    # Note for others:
    # Windows CMD: docker run --rm -p 5000:5000 -v "%cd%/models":/app/models fetch-predictor
    # Windows PowerShell: docker run --rm -p 5000:5000 -v "${pwd}/models":/app/models fetch-predictor
    ```
    *   `docker run --rm`: Same as before.
    *   `-p 5000:5000`: This is **port mapping**. It maps port 5000 on your host machine to port 5000 inside the container (where Flask is running).
        *   `HostPort:ContainerPort`
    *   `-v "$(pwd)/models":/app/models`: Same volume mount as before. This makes the `linear_regression_model.npz` file (which is now in your local `models` folder) available inside the new container at `/app/models` so `app.py` can load it.
    *   `fetch-predictor`: The image name. This time, we *don't* provide a command override, so it uses the default `CMD ["flask", "run"]` from the Dockerfile.

5.  **Access the Application:** Open your web browser (Chrome, Safari, Firefox, etc.) and go to:
    `http://localhost:5000`
    You should see the web page displaying the predicted monthly totals.

6.  **Stop the Application:** Go back to the terminal where the `docker run ... fetch-predictor` command (the one for the web app) is running. Press `Ctrl + C`. This will stop the Flask server and the container (because of `--rm`, the container will also be removed).

---

**Updated `README.md`**

# Fetch Rewards Receipt Prediction Exercise

This project predicts the approximate total number of scanned receipts per month for the year 2022, based on daily receipt data from 2021. The prediction is served via a simple Flask web application.

## Project Structure

```
fetch_receipt_prediction/
├── data/
│   └── data_daily.csv
├── models/                    # Directory for saved model artifacts
│   └── linear_regression_model.npz # Generated by train.py
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Data loading and feature engineering
│   ├── linear_regression.py   # From-scratch Linear Regression model
│   ├── prediction_logic.py    # Core prediction logic called by the app
│   └── train.py               # Script to train the model
├── templates/
│   └── index.html             # HTML template for web app
├── app.py                       # Flask web application entry point
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md                  # This file
```

## Core Logic

1.  **Data Loading (`src/data_processor.py`):** Reads daily counts from `data/data_daily.csv`, handling the specific header format.
2.  **Feature Engineering (`src/data_processor.py`):** Creates time-based features (day of year, month, day of week, time index, etc.). One-hot encoding for categoricals.
3.  **Feature Scaling (`src/data_processor.py`):** Standardizes features (mean=0, std=1) using a custom NumPy implementation. Fitted only on training data.
4.  **Model (`src/linear_regression.py`):** Linear Regression implemented from scratch using Gradient Descent (NumPy-based). Avoids high-level libraries like `scikit-learn` for the core algorithm.
5.  **Training (`src/train.py`):**
    *   Executed via `python -m src.train`.
    *   Loads 2021 data, engineers/scales features.
    *   Trains the linear regression model.
    *   Saves model weights, bias, scaler parameters, and feature column list to `models/linear_regression_model.npz`.
6.  **Prediction Logic (`src/prediction_logic.py`):**
    *   Contains the `get_monthly_predictions` function.
    *   Loads the saved model artifacts from `models/linear_regression_model.npz`.
    *   Generates dates for 2022, engineers/scales features matching training.
    *   Predicts daily counts using the model.
    *   Aggregates daily predictions into monthly totals.
    *   Returns a Pandas DataFrame with predictions and any error messages.
7.  **Inference Web App (`app.py`, `templates/index.html`):**
    *   A simple Flask web application run using `flask run`.
    *   The main route (`/`) calls `get_monthly_predictions`.
    *   Displays the predicted monthly totals for 2022 in an HTML table using the `templates/index.html` template.
    *   Shows error messages if prediction fails (e.g., model file not found).

## Setup and Usage

### Prerequisites

*   **Docker Desktop:** Installed and running ([Download Docker](https://www.docker.com/products/docker-desktop/)).
*   **Git:** For cloning the repository.
*   **Terminal/Command Prompt:** For executing commands.
*   **Web Browser:** For viewing the application.

### Option 1: Using Docker (Recommended)

This method ensures a consistent environment and handles all dependencies.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd fetch_receipt_prediction
    ```

2.  **Build the Docker Image:**
    This creates the container image containing your code and dependencies.
    ```bash
    docker build -t fetch-predictor .
    ```

3.  **Run Training (Important First Step):**
    This step runs the training script inside a temporary container to generate the `linear_regression_model.npz` file. The `-v` flag saves the output model file to your local `models` directory.
    *   **Ensure local `models` directory exists:**
        ```bash
        mkdir -p models
        ```
    *   **Run the training container (use the command appropriate for your terminal):**
        *   **macOS / Linux (bash/zsh):**
            ```bash
            docker run --rm -v "$(pwd)/models":/app/models fetch-predictor python -m src.train
            ```
        *   **Windows (Command Prompt):**
            ```bash
            docker run --rm -v "%cd%/models":/app/models fetch-predictor python -m src.train
            ```
        *   **Windows (PowerShell):**
            ```bash
            docker run --rm -v "${pwd}/models":/app/models fetch-predictor python -m src.train
            ```
    *   **Verify:** Check that `models/linear_regression_model.npz` now exists in your project folder.

4.  **Run the Web Application Container:**
    This starts the Flask application, mapping port 5000 and mounting the `models` directory so the app can load the trained model.
    *   **Run the application container (use the command appropriate for your terminal):**
        *   **macOS / Linux (bash/zsh):**
            ```bash
            docker run --rm -p 5000:5000 -v "$(pwd)/models":/app/models fetch-predictor
            ```
        *   **Windows (Command Prompt):**
            ```bash
            docker run --rm -p 5000:5000 -v "%cd%/models":/app/models fetch-predictor
            ```
        *   **Windows (PowerShell):**
            ```bash
            docker run --rm -p 5000:5000 -v "${pwd}/models":/app/models fetch-predictor
            ```

5.  **Access the App:**
    Open your web browser and navigate to: `http://localhost:5000`
    You should see the table of predicted monthly receipt counts for 2022.

6.  **Stop the App:**
    Go back to the terminal where the web application container is running (the second `docker run` command) and press `Ctrl + C`.

### Option 2: Running Locally (Without Docker)

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd fetch_receipt_prediction
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv  # Or python -m venv venv
    # Activate:
    # macOS / Linux: source venv/bin/activate
    # Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Training (Generates `models/linear_regression_model.npz`):**
    ```bash
    python -m src.train
    ```
    Ensure the model file is created in the `models` directory.

5.  **Run the Web Application:**
    ```bash
    flask run
    # Or specify host if needed: flask run --host=0.0.0.0
    ```

6.  **Access the App:**
    Open your web browser and navigate to `http://127.0.0.1:5000` (or the address shown in the terminal, usually `http://localhost:5000`).

7.  **Stop the App:**
    Press `Ctrl + C` in the terminal where `flask run` is executing.

## Design Decisions & Considerations

*   **Model Choice:** Linear Regression implemented from scratch using NumPy to demonstrate understanding of core ML principles while avoiding high-level libraries like scikit-learn for the model itself.
*   **Features:** Time-based features selected to capture potential trends and seasonality.
*   **Web Interface:** Flask provides a simple web UI for displaying predictions, fulfilling the requirement for an "app".
*   **Dockerization:** Ensures portability and consistent execution environment, simplifying setup for evaluation. Uses standard practices like volume mounting for persistent data (model file).
*   **Separation of Concerns:** Prediction logic (`prediction_logic.py`) is separated from the Flask app (`app.py`). Training (`train.py`) is a distinct script.
*   **Error Handling:** Basic error handling in the web app to show messages if the model file is missing or prediction fails.

## Future Improvements

*   **Hyperparameter Tuning:** Systematically tune the learning rate and iterations for the linear regression model.
*   **Advanced Features:** Explore lag features, rolling averages, or external regressors (e.g., holidays).
*   **More Complex Models:** Implement other models if linear regression proves insufficient (e.g., Polynomial Regression, basic time series models like ARIMA/Exponential Smoothing if library constraints allow, or models using PyTorch/TensorFlow).
*   **Cross-Validation:** Implement time-series cross-validation for more robust model evaluation during development.
*   **Visualization:** Integrate a simple chart (e.g., using Chart.js or Matplotlib served as an image) into the web app.
*   **Input:** Allow users to select the prediction year via the web interface.
*   **Deployment:** Configure a production-ready WSGI server (like Gunicorn or Waitress) within the Docker container for better performance and stability compared to the Flask development server.
```
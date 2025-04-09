FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_ENV=production 
    # Run in production mode

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#rest of the application code, data, AND PRE-TRAINED MODELS
COPY ./data ./data
COPY ./src ./src
COPY ./templates ./templates
COPY ./models ./models  
COPY app.py .


EXPOSE 5000

CMD ["flask", "run"]
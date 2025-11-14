FROM python:3.11-slim

# Optional: system deps for numpy/pandas/scikit-learn (keeps builds smoother)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Build TF-IDF artifacts (models/metadata, tfidf matrix, etc.)
RUN python scripts/build_artifacts.py

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ProximaScale — Flask app container (Person A)
# Build from PROJECT ROOT:
#   docker build -f app/Dockerfile -t proximascale-app .
#
# Run:
#   docker run -p 5000:5000 proximascale-app

FROM python:3.10-slim

WORKDIR /proximascale

# Copy only what Flask needs (not the full ML stack)
# requirements is at root; app source is in app/

COPY app/app.py app/app.py
COPY app/__init__.py app/__init__.py

# Install only Flask + psutil (skip heavy ML deps for the app container)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app/app.py"]

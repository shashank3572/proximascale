# ProximaScale — Flask app container (Person A)
# Build from the REPO ROOT (not from ./app), because the image needs both
# this app's code and the shared monitoring/ package:
#
#   docker build -f app/Dockerfile -t proximascale-app .
#
# (A build context of just ./app cannot work here — monitoring/ lives
# outside that folder and Docker cannot COPY files from outside its
# build context.)
#
# Run:
#   docker run -p 5000:5000 proximascale-app

FROM python:3.11-slim

WORKDIR /proximascale

# Install only Flask + psutil + requests (skip the full ML stack —
# this container only runs the Flask app; the metrics collector runs on the host)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py .
COPY monitoring ./monitoring

EXPOSE 5000

CMD ["python", "app.py"]
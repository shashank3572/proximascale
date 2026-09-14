# Build from the REPO ROOT (not from ./app), because the image needs both
# this app's code and the shared monitoring/ package:
#
#   docker build -f app/Dockerfile -t proximascale-app .
#
# (A build context of just ./app cannot work here — requirements.txt and
# monitoring/ live outside that folder and Docker cannot COPY files from
# outside its build context.)
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py .
COPY monitoring ./monitoring

EXPOSE 5000

CMD ["python", "app.py"]

import os

os.makedirs("deployment", exist_ok=True)

dockerfile_content = """
FROM python:3.13
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""

requirements_content = """
streamlit==1.43.2
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
huggingface_hub == 0.24.6
datasets
"""

with open("deployment/Dockerfile", "w") as f:
    f.write(dockerfile_content.strip())

with open("deployment/requirements.txt", "w") as f:
    f.write(requirements_content.strip())

print("Deployment files created successfully.")
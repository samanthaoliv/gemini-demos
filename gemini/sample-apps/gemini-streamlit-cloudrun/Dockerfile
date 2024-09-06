FROM python:3.12

EXPOSE 8080
WORKDIR /my_app

COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["streamlit", "run", "my_app.py", "--server.port=8080", "--server.address=0.0.0.0"]

import streamlit as st
from google.cloud import storage
import time


BUCKET_NAME = 'videos-news'
OUTPUT_BUCKET_NAME = 'output-news'

def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"Arquivo {source_file_name} enviado para gs://{BUCKET_NAME}/{destination_blob_name}")


    # Função para verificar se o arquivo de saída existe
def output_file_exists(output_filename):
    """Checks if the output file exists in the output bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
    blob = bucket.blob(output_filename)
    return blob.exists()

# Função para baixar o arquivo de saída
def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} baixado para {destination_file_name}.")
    
    # Interface do Streamlit
st.title("Gerador de Resumo de Vídeo")

uploaded_file = st.file_uploader("Escolha um arquivo de vídeo", type=["mp4"])

if uploaded_file is not None:
    # Faz o upload do vídeo para o bucket
    video_filename = uploaded_file.name
    upload_blob(uploaded_file, video_filename)

    # Monta o nome do arquivo de saída
    output_filename = f"resumo_{video_filename}.txt"


    # Aguarda a geração do arquivo de saída (com timeout)
    timeout = 60  # Tempo máximo de espera em segundos
    start_time = time.time()
    while not output_file_exists(output_filename):
        if time.time() - start_time > timeout:
            st.error("Tempo limite excedido. O resumo não foi gerado.")
            break
        time.sleep(1)

    if output_file_exists(output_filename):
        # Faz o download do arquivo de saída
        download_blob(output_filename, "output.txt")

        # Lê o conteúdo do arquivo e exibe no Streamlit
        with open("output.txt", "r", encoding="utf-8") as f:
            output_text = f.read()
        st.subheader("Resumo do Vídeo:")
        st.text(output_text)
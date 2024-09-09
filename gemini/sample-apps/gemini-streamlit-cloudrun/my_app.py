# pylint: disable=line-too-long,invalid-name
"""
This module demonstrates the usage of the Vertex AI Gemini 1.5 API within a Streamlit application.
"""

import os
from typing import List, Tuple, Union

from google.cloud import storage

import base64
import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_REGION")

vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models() -> Tuple[GenerativeModel, GenerativeModel]:
    """Load Gemini 1.5 Flash and Pro models."""
    return GenerativeModel("gemini-1.5-flash-001"), GenerativeModel(
        "gemini-1.5-pro-001"
    )

def download_blob_into_memory(bucket_name, blob_name):
    """Downloads a blob from the bucket into memory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()



def get_gemini_response(
    model: GenerativeModel,
    contents: Union[str, List],
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=8192
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_model_name(model: GenerativeModel) -> str:
    """Get Gemini Model Name"""
    model_name = model._model_name.replace(  # pylint: disable=protected-access
        "publishers/google/models/", ""
    )
    return f"`{model_name}`"


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


st.header("Gemini Demos Muito Legais", divider="rainbow")
gemini_15_flash, gemini_15_pro = load_models()

tab1, tab2 = st.tabs(
    ["Gerar uma historia", "Video Playground"]
)

with tab1:
    st.subheader("Generate a story")

    selected_model = st.radio(
        "Select Gemini Model:",
        [gemini_15_flash, gemini_15_pro],
        format_func=get_model_name,
        key="selected_model_story",
        horizontal=True,
    )

    # Story premise
    character_name = st.text_input(
        "Qual o nome do personagem?: \n\n", key="character_name", value="Darth"
    )
    character_type = st.text_input(
        "Qual o tipo do seu peresonagem? \n\n", key="character_type", value="Gato"
    )
    character_persona = st.text_input(
        "Qual a personalidade do seu personagem? \n\n",
        key="character_persona",
        value="Darth e um gatinho muito carinhoso.",
    )
    character_location = st.text_input(
        "Onde o personagem mora? \n\n",
        key="character_location",
        value="Belem/PA",
    )
    story_premise = st.multiselect(
        "Qual a premissa da sua historia? (pode escolher mais de uma opcao) \n\n",
        [
            "Amor",
            "Aventura",
            "Misterio",
            "Comedia",
            "Sci-Fi",
            "Fantasia",
            "Terror",
        ],
        key="story_premise",
        default=["Amor", "Aventura"],
    )
    creative_control = st.radio(
        "Escolha um nivel de criatividade: \n\n",
        ["Low", "High"],
        key="creative_control",
        horizontal=True,
    )
    length_of_story = st.radio(
        "Escolha o tamanho da sua historia: \n\n",
        ["Curta", "Longa"],
        key="length_of_story",
        horizontal=True,
    )

    if creative_control == "Low":
        temperature = 0.30
    else:
        temperature = 0.95

    if length_of_story == "Curta":
        max_output_tokens = 2048
    else:
        max_output_tokens = 8192

    prompt = f"""Escreva uma historia {length_of_story} baseada na premissa a seguir: \n
    nome do personagem: {character_name} \n
    tipo do personagem: {character_type} \n
    personalidade do personagem: {character_persona} \n
    localizacao do personagem: {character_location} \n
    premissa da historia: {",".join(story_premise)} \n
    Se a historia for "curta", certifique-se de ter 5 capitulos ou entao se for "longa", entao 10 capitulos.
    O ponto importante e que cada capitulo deve ser gerado com base na premissa dada acima.
    Primeiro comece dando a introducao do livro, introducoes dos capitulos e entao cada capitulo. Ele tambem deve ter um final apropriado.
    O livro deve ter prologo e epilogo.
    """
    config = GenerationConfig(
        temperature=temperature, max_output_tokens=max_output_tokens
    )

    generate_t2t = st.button("Gerar uma historia", key="generate_t2t")
    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner(
            f"Gerando a sua historia usando {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
            with first_tab1:
                response = get_gemini_response(
                    selected_model,  # Use the selected model
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Sua historia:")
                    st.write(response)
            with first_tab2:
                st.text(
                    f"""Parameters:\n- Temperature: {temperature}\n- Max Output Tokens: {max_output_tokens}\n"""
                )
                st.text(prompt)


with tab2:
    st.subheader("Video Discription")

    selected_model = st.radio(
        "Select Gemini Model:",
        [gemini_15_flash, gemini_15_pro],
        format_func=get_model_name,
        key="selected_model_video",
        horizontal=True,
    )

    video_desc_tab = st.tabs(["Descrição do Vídeo"])[0]

    with video_desc_tab:
        st.markdown("""Gemini 1.5 Pro pode criar um texto a partir do vídeo:""")


    video1 = Part.from_uri(
        mime_type="video/mp4",
        uri="gs://news-videofiles/news/Adolescente filha de brasileiros está desaparecida em Nova Jersey.mp4")
        
        # Download video data
    bucket_name = "news-videofiles"
    blob_name = "news/Adolescente filha de brasileiros está desaparecida em Nova Jersey.mp4"
    video_data = download_blob_into_memory(bucket_name, blob_name)

    st.video(video_data)

    transcription_prompt = f"""Transcreva o seguinte vídeo em português: {video1}"""

    try:
        with st.spinner("Gerando transcrição do vídeo usando Gemini..."):
            transcript_text = get_gemini_response(selected_model, [transcription_prompt])

        prompt = f"""Faça um resumo do seguinte vídeo, como se fosse para um site de notícias:

            **Transcrição do vídeo:** {transcript_text}

            - Quem são as pessoas envolvidas?
            - Onde aconteceu?
            """

        response_tab, prompt_tab = st.tabs(["Resposta", "Prompt"])
        vide_desc_description = st.button("Gerar descrição do vídeo", key="vide_desc_description")

        if vide_desc_description:
            with st.spinner(f"Gerando descrição do vídeo usando {get_model_name(selected_model)} ..."):
                response = get_gemini_response(selected_model, [prompt])

                with response_tab:
                    st.markdown(response)
                    st.markdown("\n\n\n")

    except Exception as e:
        st.error(f"Erro ao processar o vídeo: {e}")
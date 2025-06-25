import streamlit as st
import sys
import os
import logging
import threading
import json
import configparser
from datetime import datetime
from TTS.api import TTS
from gtts import gTTS
import pyttsx3
import simpleaudio as sa

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.prompts.base import PromptTemplate
from utils import get_automerging_query_engine, get_sentence_window_query_engine

# Set up logging
config = configparser.ConfigParser()
config.read('config.ini')

indextype = config['api']['indextype']
embed_modelname = config['api']['embedmodel']
basic_idx_dir = config['index']['basic_idx_dir']
sent_win_idx_dir = config['index']['sent_win_idx_dir']
auto_mrg_idx_dir = config['index']['auto_mrg_idx_dir']
ttsengine = config['api']['ttsengine']
useopenai = config.getboolean('api', 'useopenai')
openai_api_base = config['api']['openai_api_base']

log_level_str = config.get('api', 'loglevel', fallback='WARNING').upper()
log_level = getattr(logging, log_level_str, logging.WARNING)
logging.basicConfig(stream=sys.stdout, level=log_level)

# Optional: Set up LLM
from langchain_community.llms import LlamaCpp
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

stop_words = config.get("Prompt", "stop_words", fallback="").split(", ")
modelname = config['api']['local_modelname']

set_llm_cache(InMemoryCache())

llm = LlamaCpp(
    model_path="./models/" + modelname,
    cache=True,
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=2048,
    n_threads=8,
    temperature=0.01,
    max_tokens=512,
    f16_kv=True,
    repeat_penalty=1.1,
    top_p=0.95,
    top_k=40,
    stop=stop_words
)

Settings.llm = llm
Settings.embed_model = embed_modelname

# TTS setup
if ttsengine == 'coqui':
    tts = TTS(model_name="tts_models/en/ljspeech/vits--neon", progress_bar=False).to("cuda")
elif ttsengine == 'gtts':
    tts = gTTS(text='Hello', lang='en')
else:
    tts = pyttsx3.init()
    voices = tts.getProperty('voices')
    tts.setProperty('voice', voices[1].id)
    rate = tts.getProperty('rate')
    tts.setProperty('rate', rate - 50)

# Load index
index_directory = {
    'basic': basic_idx_dir,
    'sentence': sent_win_idx_dir,
    'automerge': auto_mrg_idx_dir
}[indextype]

storage_context = StorageContext.from_defaults(persist_dir=index_directory)
index = load_index_from_storage(storage_context=storage_context)

if indextype == 'basic':
    query_engine = index.as_query_engine()
elif indextype == 'sentence':
    query_engine = get_sentence_window_query_engine(index)
else:
    query_engine = get_automerging_query_engine(automerging_index=index)

# Update prompts
qa_prompt_tmpl_str = config.get("Prompt", "qa_prompt_tmpl", fallback="").strip()
qa_prompt_tmpl_str = qa_prompt_tmpl_str.replace("{current_date}", datetime.now().strftime('%d %B %Y'))
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})


# Helper
def play_sound_then_delete(path_to_wav):
    def play_and_delete():
        try:
            wave_obj = sa.WaveObject.from_wave_file(path_to_wav)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            try:
                os.remove(path_to_wav)
            except Exception as e:
                print(f"Error deleting file: {e}")

    threading.Thread(target=play_and_delete, daemon=True).start()


# Streamlit UI
st.title("Chat with your data")

input_text = st.text_area("Enter your text", height=200)
generate_audio = st.checkbox("Generate audio response")

if st.button("Submit") and input_text.strip():
    response = query_engine.query(input_text)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    st.markdown(f"**Response:**\n\n{response.response}")

    if generate_audio:
        output_path = f"temp_output_{timestamp}.wav"
        if ttsengine == 'coqui':
            tts.tts_to_file(text=response.response, file_path=output_path)
        elif ttsengine == 'gtts':
            tts = gTTS(text=response.response, lang='en')
            tts.save(output_path)
        else:
            tts.save_to_file(response.response, output_path)
            tts.runAndWait()
        st.audio(output_path)
        play_sound_then_delete(output_path)

import io
import json
import zipfile
import threading
import queue
import logging
import time
from pathlib import Path

import streamlit as st
import yaml

# CONFIG UI
st.set_page_config(page_title="ELOQUENT Pipeline", layout="wide")
st.title("ELOQUENT — Cultural Robustness & Diversity")

# SESSION STATE
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

if "thread" not in st.session_state:
    st.session_state.thread = None

if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()

if "logs" not in st.session_state:
    st.session_state.logs = []

def is_running():
    t = st.session_state.thread
    return t is not None and t.is_alive()

# CONSTANTES
LANGUAGES_AVAILABLE = {
    "Français (fr)": "fr",
    "Anglais (en)": "en",
    "Espagnol (es)": "es",
}

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CONFIG_DIR = Path("config")

# UTILS
def clean_name(name):
    return name.replace(":", "_").replace("/", "_")

def get_available_languages():
    return [
        label for label, code in LANGUAGES_AVAILABLE.items()
        if (DATA_DIR / f"{code}_specific.jsonl").exists()
    ]

def count_answered(filepath):
    if not filepath.exists():
        return 0
    with open(filepath, encoding="utf-8") as f:
        return sum(
            1 for l in f
            if l.strip() and json.loads(l).get("answer")
        )

# CONFIG BUILD
def build_config(provider, groq_model, local_model, languages,
                 dataset_type, temperature, max_tokens, delay, variant):

    model_name = groq_model if provider == "groq" else local_model

    return {
        "provider": provider,
        "model_name": clean_name(model_name),
        "languages": languages,
        "dataset_type": dataset_type,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "delay": delay,
        "variant": variant,
        "data_dir": "data",
        "output_dir": "outputs",
    }

def save_config(config):
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / "run_temp.yaml"
    yaml.dump(config, open(path, "w", encoding="utf-8"))
    return str(path)

# PIPELINE THREAD
def run_pipeline(config_path, log_queue, stop_event):

    logger = logging.getLogger("pipeline")
    logger.handlers = []

    class QH(logging.Handler):
        def emit(self, record):
            log_queue.put(self.format(record))

    handler = QH()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        config = yaml.safe_load(open(config_path))

        for lang in config["languages"]:

            input_file = DATA_DIR / f"{lang}_{config['dataset_type']}.jsonl"
            output_file = OUTPUT_DIR / f"{lang}_{config['dataset_type']}_{config['variant']}_{config['model_name']}.jsonl"

            output_file.parent.mkdir(exist_ok=True)

            already_done = count_answered(output_file)
            logger.info(f"{lang} → reprise à {already_done}")

            with open(input_file, encoding="utf-8") as f_in, \
                 open(output_file, "a", encoding="utf-8") as f_out:

                for i, line in enumerate(f_in):

                    if stop_event.is_set():
                        logger.info("Arrêt demandé")
                        return

                    if i < already_done:
                        continue

                    data = json.loads(line)

                    time.sleep(config["delay"])
                    data["answer"] = f"Réponse ({config['variant']})"

                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

                    logger.info(f"OK — {lang} #{i}")

        log_queue.put("__DONE__")

    except Exception as e:
        log_queue.put(f"ERREUR: {e}")
        log_queue.put("__DONE__")

# SIDEBAR
with st.sidebar:

    provider = st.selectbox("Provider", ["groq", "local"])
    groq_model = st.text_input("Groq model", "llama-3.3")
    local_model = st.text_input("Local model", "gemma4:31b")

    variant = st.selectbox("Variant", [
        "baseline", "system_constrained", "cot_cultural", "rewritten_query"
    ])

    dataset_type = st.radio("Dataset", ["specific", "unspecific"])

    selected = st.multiselect("Langues", get_available_languages())
    selected_codes = [LANGUAGES_AVAILABLE[l] for l in selected]

    temperature = st.slider("Température", 0.0, 1.0, 0.0)
    max_tokens = st.number_input("Max tokens", 10, 500, 100)
    delay = st.slider("Delay", 0.0, 3.0, 0.5)

# BUTTONS
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    launch = st.button("Lancer", type="primary", use_container_width=True)
    stop = st.button("Arrêter", use_container_width=True)

# START RUN
if launch:

    if is_running():
        st.warning("Run déjà en cours")

    elif not selected_codes:
        st.warning("Sélectionne au moins une langue")

    else:
        # reset COMPLET du run (CRUCIAL)
        st.session_state.stop_event = threading.Event()  # évite STOP fantôme
        st.session_state.logs = []  # reset console
        st.session_state.log_queue = queue.Queue()  # reset logs

        config = build_config(
            provider, groq_model, local_model, selected_codes,
            dataset_type, temperature, max_tokens, delay, variant
        )

        path = save_config(config)

        thread = threading.Thread(
            target=run_pipeline,
            args=(path, st.session_state.log_queue, st.session_state.stop_event),
            daemon=True
        )

        thread.start()
        st.session_state.thread = thread

# STOP
if stop and is_running():
    st.session_state.stop_event.set()
    st.info("Arrêt en cours...")

# LOG CONSOLE
log_container = st.container(height=450, border=True)

with log_container:

    # affichage du run courant
    while is_running() or not st.session_state.log_queue.empty():

        try:
            msg = st.session_state.log_queue.get_nowait()

            if msg == "__DONE__":
                break

            st.session_state.logs.append(msg)

        except queue.Empty:
            pass

        # limite mémoire console
        st.session_state.logs = st.session_state.logs[-300:]

        # affichage console stable (ne scroll pas la page)
        st.code("\n".join(st.session_state.logs))

        time.sleep(0.15)

    if not is_running():
        st.success("Run terminé")

# EXPORT
st.divider()

if OUTPUT_DIR.exists():

    files = list(OUTPUT_DIR.glob("*.jsonl"))

    for f in files:
        st.write(f"{f.name} — {count_answered(f)} réponses")

    if files:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f in files:
                z.write(f, f.name)

        buf.seek(0)

        st.download_button("Télécharger ZIP", buf)
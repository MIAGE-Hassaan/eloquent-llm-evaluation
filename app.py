import io
import json
import zipfile
import threading
import queue
import logging
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import yaml

# Config page
st.set_page_config(page_title="ELOQUENT Pipeline", layout="wide")
st.title("ELOQUENT — Cultural Robustness & Diversity")

# Initialisation du state Streamlit
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "thread" not in st.session_state:
    st.session_state.thread = None
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if "logs" not in st.session_state:
    st.session_state.logs = []

# Constantes
LANGUAGES_AVAILABLE = {
    "Français (fr)": "fr",
    "Anglais (en)": "en",
    "Espagnol (es)": "es",
    "Allemand (de)": "de",
    "Russe (ru)": "ru",
}

DATA_DIR        = Path("data")
OUTPUT_DIR      = Path("outputs")
CONFIG_DIR      = Path("config")
BASELINE_CONFIG = CONFIG_DIR / "baseline.yaml"

# Helpers
def is_running():
    t = st.session_state.thread
    return t is not None and t.is_alive()


def clean_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(".", "_")


def get_available_languages():
    return [
        label for label, code in LANGUAGES_AVAILABLE.items()
        if (DATA_DIR / f"{code}_specific.jsonl").exists()
        or (DATA_DIR / f"{code}_unspecific.jsonl").exists()
    ]


def count_answered(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    if json.loads(line).get("answer"):
                        count += 1
                except Exception:
                    pass
    return count


def load_baseline() -> dict:
    if BASELINE_CONFIG.exists():
        with open(BASELINE_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# Config
def build_config(provider, groq_model, local_model, languages,
                 dataset_type, temperature, max_tokens, delay, variant) -> dict:
    config = load_baseline()

    config["provider"]      = provider
    config["groq_model"]    = groq_model
    config["local_model"]   = local_model
    config["model_name"]    = clean_name(groq_model if provider == "groq" else local_model)
    config["languages"]     = languages
    config["dataset_type"]  = dataset_type
    config["temperature"]   = temperature
    config["max_tokens"]    = max_tokens
    config["delay_seconds"] = delay   # clé lue par run_language dans pipeline.py
    config["delay"]         = delay   # conservé pour compatibilité run_pipeline_logic
    config["variant"]       = variant
    config["data_dir"]      = "data"
    config["output_dir"]    = "outputs"

    return config


def save_config(config: dict) -> str:
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / "run_temp.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    return str(path)

# Génération du fichier metadata — conforme au format ELOQUENT
def save_submission_metadata(config: dict, timestamp: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / f"submission_metadata_{timestamp}.json"

    # Nom du modèle selon le provider
    model_name = (
        config.get("groq_model", "unknown")
        if config.get("provider") == "groq"
        else config.get("local_model", "unknown")
    )

    # Variant actif et ses templates
    variant_active = config.get("variant", "baseline")
    templates      = config.get("variants_templates", {})
    full_template  = templates.get(variant_active, "{question}")

    # Découpage prefix / suffix autour de {question}
    if "{question}" in full_template:
        parts         = full_template.split("{question}", 1)
        prompt_prefix = parts[0].strip()
        prompt_suffix = parts[1].strip()
    else:
        prompt_prefix = full_template.strip()
        prompt_suffix = ""

    metadata = {
        "team":         "votre-nom-de-groupe",
        "system":       "eloquent-miage-v1",
        "model":        model_name,
        "submissionid": f"experiment-{timestamp}",
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "label":        "eloquent-2026-cultural",
        "languages":    config.get("languages", []),
        "modifications": {
            # system_prompt : template complet si variant actif, sinon neutre
            "system_prompt":         full_template if variant_active != "baseline" else "Standard LLM prompt",
            "prompt_prefix_generic": prompt_prefix,
            "prompt_suffix_generic": prompt_suffix,
            "generation_params": {
                "do_sample":      config.get("temperature", 0) > 0,
                "max_new_tokens": config.get("max_tokens", 200),
                "temperature":    config.get("temperature", 0),
            },
            "notes": (
                f"Test de la variante '{variant_active}' "
                f"sur dataset '{config.get('dataset_type', 'unknown')}' "
                f"avec {config.get('provider', 'unknown')}"
            ),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    return metadata_path

# Thread du pipeline
def run_pipeline_logic(config_path: str, log_queue: queue.Queue, stop_event: threading.Event):
    logger = logging.getLogger("pipeline")
    logger.handlers = []

    class QH(logging.Handler):
        def emit(self, record):
            log_queue.put(self.format(record))

    handler = QH()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        for lang in config["languages"]:
            input_file = DATA_DIR / f"{lang}_{config['dataset_type']}.jsonl"
            out_name   = f"{lang}_{config['dataset_type']}_{config['variant']}_{config['model_name']}.jsonl"
            output_file = OUTPUT_DIR / out_name

            OUTPUT_DIR.mkdir(exist_ok=True)

            already_done = count_answered(output_file)
            logger.info(f"{lang} ({config['variant']}) → reprise à {already_done}")

            if not input_file.exists():
                logger.error(f"Fichier source introuvable : {input_file}")
                continue

            with open(input_file, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()

            with open(output_file, "a", encoding="utf-8") as f_out:
                for i, line in enumerate(lines):
                    if stop_event.is_set():
                        logger.info("Arrêt manuel détecté")
                        raise InterruptedError("Arrêt manuel")

                    if i < already_done:
                        continue

                    data = json.loads(line)
                    time.sleep(config.get("delay", 0))

                    # ← remplacer par l'appel réel au prompter
                    data["answer"]  = f"Réponse simulée pour le variant {config['variant']}"
                    data["variant"] = config["variant"]

                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    logger.info(f"OK — {lang} question #{i+1}")

        log_queue.put("__DONE__")

    except InterruptedError:
        log_queue.put("Run interrompu manuellement.")
        log_queue.put("__DONE__")

    except Exception as e:
        log_queue.put(f"ERREUR CRITIQUE : {e}")
        log_queue.put("__DONE__")

    finally:
        # S'exécute toujours : run normal, stop bouton, ou exception
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            metadata_path = save_submission_metadata(config, timestamp)
            logger.info(f"✅ metadata sauvegardé → {metadata_path.name}")
        except Exception as e:
            log_queue.put(f"⚠️ Echec écriture metadata : {e}")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    provider    = st.selectbox("Provider", ["groq", "local"])
    groq_model  = st.text_input("Groq model", "llama-3.3")
    local_model = st.text_input("Local model", "gemma2:9b")

    st.divider()

    variant = st.selectbox("Variant (Lot C)", [
        "baseline", "system_constrained", "cot_cultural", "rewritten_query"
    ])
    dataset_type = st.radio("Dataset", ["specific", "unspecific"])

    available      = get_available_languages()
    selected       = st.multiselect("Langues", available, default=available[:1])
    selected_codes = [LANGUAGES_AVAILABLE[l] for l in selected]

    st.divider()
    temperature = st.slider("Température", 0.0, 1.0, 0.0)
    max_tokens  = st.number_input("Max tokens", 10, 500, 100)
    delay       = st.slider("Délai (s)", 0.0, 3.0, 0.5)

# Boutons
c1, c2, c3 = st.columns(3)
with c1:
    launch = st.button("Lancer le run", type="primary", use_container_width=True)
with c2:
    stop = st.button("Arrêter", use_container_width=True)

if launch:
    if is_running():
        st.warning("Un pipeline est déjà en cours d'exécution.")
    elif not selected_codes:
        st.error("Veuillez sélectionner au moins une langue.")
    else:
        st.session_state.stop_event.clear()
        st.session_state.logs = []
        st.session_state.log_queue = queue.Queue()

        cfg      = build_config(provider, groq_model, local_model, selected_codes,
                                dataset_type, temperature, max_tokens, delay, variant)
        cfg_path = save_config(cfg)

        t = threading.Thread(
            target=run_pipeline_logic,
            args=(cfg_path, st.session_state.log_queue, st.session_state.stop_event),
            daemon=True,
        )
        t.start()
        st.session_state.thread = t
        st.rerun()

if stop and is_running():
    st.session_state.stop_event.set()
    st.info("Signal d'arrêt envoyé.")

# Console
st.subheader("Console de progression")
log_container = st.empty()

if is_running() or st.session_state.logs:
    while True:
        while not st.session_state.log_queue.empty():
            msg = st.session_state.log_queue.get()
            if msg == "__DONE__":
                break
            st.session_state.logs.append(msg)

        st.session_state.logs = st.session_state.logs[-15:]
        log_container.code("\n".join(st.session_state.logs))

        if not is_running():
            break
        time.sleep(0.1)

# Export
st.divider()
st.subheader("📦 Export des données")

if OUTPUT_DIR.exists():
    all_files = list(OUTPUT_DIR.glob("*.*"))

    if all_files:
        for f in sorted(all_files):
            if f.suffix == ".jsonl":
                done = count_answered(f)
                st.write(f"`{f.name}` : **{done}** réponses générées")
            elif f.suffix == ".json":
                st.write(f"`{f.name}` (Métadonnées Challenge ELOQUENT)")
            elif f.suffix == ".yaml":
                st.write(f"`{f.name}` (Configuration du run)")

        st.write("---")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in all_files:
                z.write(f, f.name)

        st.download_button(
            label="Télécharger le package complet (ZIP)",
            data=buf.getvalue(),
            file_name=f"eloquent_submission_{datetime.now().strftime('%d%m_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.info("Le dossier d'export est vide. Lancez un run pour générer des données.")
else:
    st.info("Aucun résultat disponible pour le moment.")

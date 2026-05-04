import io
import json
import logging
import queue
import threading
import time
import zipfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ELOQUENT Pipeline", layout="wide")
st.title("ELOQUENT — Cultural Robustness & Diversity")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "stop_event": threading.Event(),
    "thread":     None,
    "log_queue":  queue.Queue(),
    "logs":       [],
    "run_done":   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Constantes — 22 langues disponibles sur la plateforme ELOQUENT
# ---------------------------------------------------------------------------
LANGUAGES_AVAILABLE = {
    "Anglais (en)":    "en",
    "Français (fr)":   "fr",
    "Espagnol (es)":   "es",
    "Allemand (de)":   "de",
    "Russe (ru)":      "ru",
    "Bengali (bn)":    "bn",
    "Catalan (ca)":    "ca",
    "Tchèque (cs)":    "cs",
    "Danois (da)":     "da",
    "Féroïen (fo)":    "fo",
    "Finnois (fi)":    "fi",
    "Grec (el)":       "el",
    "Hébreu (he)":     "he",
    "Hindi (hi)":      "hi",
    "Italien (it)":    "it",
    "Kannada (kn)":    "kn",
    "Marathi (mr)":    "mr",
    "Polonais (pl)":   "pl",
    "Slovaque (sk)":   "sk",
    "Suédois (sv)":    "sv",
    "Tamoul (ta)":     "ta",
    "Télougou (te)":   "te",
}

DATA_DIR        = Path("data")
OUTPUT_DIR      = Path("outputs")
CONFIG_DIR      = Path("config")
BASELINE_CONFIG = CONFIG_DIR / "baseline.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_running() -> bool:
    t = st.session_state.thread
    return t is not None and t.is_alive()


def clean_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace(".", "_")


def get_available_languages() -> list[str]:
    """Retourne les langues dont au moins un fichier data existe."""
    return [
        label for label, code in LANGUAGES_AVAILABLE.items()
        if (DATA_DIR / f"{code}_specific.jsonl").exists()
        or (DATA_DIR / f"{code}_unspecific.jsonl").exists()
    ]


def count_answered(filepath: Path) -> int:
    """Compte les réponses valides (hors simulées)."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ans = json.loads(line).get("answer") or ""
                    if ans and "simul" not in ans.lower():
                        count += 1
                except Exception:
                    pass
    return count


def count_simulated(filepath: Path) -> int:
    """Compte les réponses simulées (anciennes, invalides)."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ans = json.loads(line).get("answer") or ""
                    if ans and "simul" in ans.lower():
                        count += 1
                except Exception:
                    pass
    return count


def load_baseline() -> dict:
    if BASELINE_CONFIG.exists():
        with open(BASELINE_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

# ---------------------------------------------------------------------------
# Construction de la config runtime
# ---------------------------------------------------------------------------

def build_config(provider, groq_model, local_model, languages,
                 dataset_type, temperature, max_tokens, delay, variant) -> dict:
    config = load_baseline()                    # base = baseline.yaml
    config["provider"]      = provider
    config["groq_model"]    = groq_model
    config["local_model"]   = local_model
    config["model_name"]    = clean_name(groq_model if provider == "groq" else local_model)
    config["languages"]     = languages
    config["dataset_type"]  = dataset_type
    config["temperature"]   = temperature
    config["max_tokens"]    = max_tokens
    config["delay_seconds"] = delay
    config["variant"]       = variant
    config["data_dir"]      = "data"
    config["output_dir"]    = "outputs"
    config["log_dir"]       = "logs"
    return config


def save_config(config: dict) -> str:
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / "run_temp.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    return str(path)

# ---------------------------------------------------------------------------
# Thread du pipeline — appelle le VRAI run_pipeline()
# ---------------------------------------------------------------------------

def run_pipeline_thread(config_path: str, log_queue: queue.Queue,
                        stop_event: threading.Event):
    """
    Exécuté dans un thread séparé.
    Injecte un QueueHandler sur le logger 'pipeline' AVANT d'appeler
    run_pipeline() — setup_logger() ne réinitialisera pas le logger
    puisque handlers sera déjà rempli.
    """
    import pipeline as pl

    logger = logging.getLogger("pipeline")
    logger.handlers = []          # réinitialise pour ce run

    class _QueueHandler(logging.Handler):
        def emit(self, record):
            log_queue.put(self.format(record))

    handler = _QueueHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        pl.run_pipeline(config_path, stop_event=stop_event)
    except Exception as exc:
        log_queue.put(f"ERREUR CRITIQUE : {exc}")
    finally:
        log_queue.put("__DONE__")

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    provider    = st.selectbox("Provider", ["groq", "local"])
    groq_model  = st.text_input("Groq model",  "llama-3.1-8b-instant")
    local_model = st.text_input("Local model", "gemma4:31b")

    st.divider()

    variant = st.selectbox("Variante", [
        "baseline", "system_constrained", "cot_cultural", "rewritten_query"
    ], help=(
        "baseline : prompt brut\n"
        "system_constrained : system prompt culturel\n"
        "cot_cultural : raisonnement chain-of-thought\n"
        "rewritten_query : reformulation de la question"
    ))
    dataset_type = st.radio("Dataset", ["specific", "unspecific"])

    available      = get_available_languages()
    selected       = st.multiselect("Langues", available, default=available[:2])
    selected_codes = [LANGUAGES_AVAILABLE[l] for l in selected]

    st.divider()
    temperature = st.slider("Température", 0.0, 1.0, 0.0,
                            help="0 = déterministe (obligatoire pour la baseline)")
    max_tokens  = st.number_input("Max tokens", 10, 500, 200,
                                  help="Spec ELOQUENT : max_new_tokens=200")
    delay       = st.slider("Délai entre questions (s)", 0.0, 3.0, 0.5)

# ---------------------------------------------------------------------------
# Boutons de contrôle
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    launch = st.button("Lancer le run", type="primary", use_container_width=True)
with c2:
    stop = st.button("Arrêter", use_container_width=True)

if launch:
    if is_running():
        st.warning("Un pipeline est déjà en cours d'exécution.")
    elif not selected_codes:
        st.error("Sélectionnez au moins une langue.")
    else:
        st.session_state.stop_event = threading.Event()
        st.session_state.logs       = []
        st.session_state.log_queue  = queue.Queue()
        st.session_state.run_done   = False

        cfg      = build_config(provider, groq_model, local_model, selected_codes,
                                dataset_type, temperature, max_tokens, delay, variant)
        cfg_path = save_config(cfg)

        t = threading.Thread(
            target=run_pipeline_thread,
            args=(cfg_path, st.session_state.log_queue, st.session_state.stop_event),
            daemon=True,
        )
        t.start()
        st.session_state.thread = t
        st.rerun()

if stop and is_running():
    st.session_state.stop_event.set()
    st.info("Signal d'arrêt envoyé.")

# ---------------------------------------------------------------------------
# Console de logs — polling non-bloquant via st.rerun()
# ---------------------------------------------------------------------------
st.subheader("Console de progression")
log_box = st.empty()

# Drain de la queue à chaque rendu
while not st.session_state.log_queue.empty():
    try:
        msg = st.session_state.log_queue.get_nowait()
        if msg == "__DONE__":
            st.session_state.run_done = True
        else:
            st.session_state.logs.append(msg)
    except queue.Empty:
        break

if st.session_state.logs:
    log_box.code("\n".join(st.session_state.logs[-25:]))
elif not is_running():
    log_box.info("Lancez un run pour voir la progression ici.")

# Si le pipeline tourne toujours, on se re-render dans 1s
if is_running():
    time.sleep(1)
    st.rerun()
elif st.session_state.run_done:
    st.success("✅ Run terminé ! Consultez la section Export ci-dessous.")
    st.session_state.run_done = False

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📦 Export des données")

# Tous les fichiers : outputs/ racine + submission/ racine
all_files = []
if OUTPUT_DIR.exists():
    all_files += sorted(f for f in OUTPUT_DIR.glob("*.*") if f.is_file())
submission_dir = Path("submission")
if submission_dir.exists():
    all_files += sorted(f for f in submission_dir.glob("*.*") if f.is_file())

if all_files:
    for f in all_files:
        if f.suffix == ".jsonl":
            done = count_answered(f)
            simul = count_simulated(f)
            label = "(soumission)" if f.parent.name == "submission" else ""
            msg = f"`{f.name}` {label}: **{done}** réponses"
            if simul > 0:
                msg += f" (⚠️ {simul} simulées)"
            st.write(msg)
        elif f.suffix == ".json":
            label = "(soumission)" if f.parent.name == "submission" else ""
            st.write(f"`{f.name}` {label} (Métadonnées ELOQUENT)")
        elif f.suffix == ".yaml":
            st.write(f"`{f.name}` (Configuration du run)")

    st.write("---")

# ZIP de soumission par expérience
submission_dir = Path("submission")
if submission_dir.exists():
    experiments = sorted([d for d in submission_dir.iterdir() if d.is_dir()], reverse=True)
    if experiments:
        st.write("---")
        st.write("📑 **Packages de soumission par expérience**")
        for exp_path in experiments:
            exp_name = exp_path.name
            zip_files = list(exp_path.glob("*.*"))
            
            col_info, col_dl = st.columns([3, 1])
            with col_info:
                st.write(f"📁 `{exp_name}` ({len(zip_files)} fichiers)")
            with col_dl:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in zip_files:
                        z.write(f, f.name)
                
                st.download_button(
                    label="ZIP",
                    data=buf.getvalue(),
                    file_name=f"eloquent_submission_{exp_name}.zip",
                    mime="application/zip",
                    key=f"dl_{exp_name}"
                )
else:
    st.info("Aucune soumission prête. Lancez un run pour générer des fichiers dans `submission/`.")

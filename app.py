import io
import json
import zipfile
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st
import yaml

# Configuration de la page
st.set_page_config(
    page_title="ELOQUENT Pipeline",
    layout="wide",
)

# Masquer la navbar Streamlit
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Constantes 
# Correspondance nom affiché → code langue
LANGUAGES_AVAILABLE = {
    "Français (fr)": "fr",
    "Anglais (en)": "en",
    "Espagnol (es)": "es",
    "Allemand (de)": "de",
    "Russe (ru)": "ru",
}

DATA_DIR    = Path("data")
OUTPUT_DIR  = Path("outputs")
CONFIG_DIR  = Path("config")

# Fonctions utilitaires
def get_available_languages():
    # Retourne uniquement les langues dont le fichier JSONL existe dans data/
    return [
        label for label, code in LANGUAGES_AVAILABLE.items()
        if (DATA_DIR / f"{code}_specific.jsonl").exists()
        or (DATA_DIR / f"{code}_unspecific.jsonl").exists()
    ]


def count_lines(filepath):
    # Compte le nombre de lignes non vides dans un fichier JSONL
    if not filepath.exists():
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_answered(filepath):
    #Compte les entrées qui ont déjà un champ 'answer' non nul
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    if json.loads(line).get("answer") is not None:
                        count += 1
                except Exception:
                    pass
    return count


def build_config(provider, groq_model, local_model, languages, dataset_type,
                 temperature, max_tokens, delay, max_questions):
    
    # Construit le dictionnaire de configuration à partir des choix de l'interface.
    # Ce dict est ensuite sauvegardé en YAML et passé à run_pipeline().
    config = {
        "provider": provider,
        "groq_model": groq_model,
        "local_model": local_model,
        "local_base_url": "http://localhost:11434",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "delay_seconds": delay,
        "dataset_type": dataset_type,
        "languages": languages,
        "data_dir": "data",
        "output_dir": "outputs",
        "log_dir": "logs",
    }
    # max_questions = 0 signifie "pas de limite"
    if max_questions > 0:
        config["max_questions"] = max_questions
    return config


def save_temp_config(config):
    #Sauvegarde la config dans config/run_temp.yaml et retourne son chemin
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / "run_temp.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    return str(path)


def make_zip():
    
    # Crée un fichier zip en mémoire contenant tous les JSONL et YAML du dossier outputs/
    # Retourne les bytes du zip, prêt pour st.download_button
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.glob("*.jsonl"):
            zf.write(f, f"outputs/{f.name}")
        for f in OUTPUT_DIR.glob("*.yaml"):
            zf.write(f, f"outputs/{f.name}")
    buf.seek(0)
    return buf.read()


# Exécution du pipeline dans un thread séparé

def run_pipeline_threaded(config_path, log_queue):
    # Lance run_pipeline() dans un thread séparé pour ne pas bloquer l'interface.
    # Redirection des logs du pipeline vers la queue
    class QueueHandler(logging.Handler):
        def emit(self, record):
            log_queue.put(self.format(record))

    # Réinitialisation des handlers pour éviter les doublons entre runs
    logger = logging.getLogger("pipeline")
    logger.handlers = []
    handler = QueueHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        from pipeline import run_pipeline
        run_pipeline(config_path)
        log_queue.put("__DONE__")
    except Exception as e:
        log_queue.put(f"[ERREUR CRITIQUE] {e}")
        log_queue.put("__DONE__")


# Interface principale 

st.title("ELOQUENT — Cultural Robustness & Diversity")
st.caption("Pipeline multilingue pour le challenge CLEF 2026")


# Sidebar : tous les paramètres du run
with st.sidebar:
    st.header("Configuration du run")

    # Choix du provider
    provider = st.selectbox("Provider", ["groq", "local"])

    if provider == "groq":
        groq_model = st.selectbox("Modèle Groq", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ])
        local_model = "llama3"  # valeur par défaut, non utilisée
    else:
        local_model = st.text_input("Modèle local (Ollama)", value="gemma4:31b")
        groq_model = "llama-3.3-70b-versatile"  # valeur par défaut, non utilisée

    st.divider()

    # Choix du dataset
    dataset_type = st.radio("Type de dataset", ["specific", "unspecific"])

    # Choix des langues — seules celles présentes dans data/ sont proposées
    available_langs = get_available_languages()
    if available_langs:
        selected_labels = st.multiselect(
            "Langues",
            options=available_langs,
            default=available_langs[:1],
        )
        selected_codes = [LANGUAGES_AVAILABLE[l] for l in selected_labels]
    else:
        st.warning("Aucun fichier JSONL trouvé dans data/")
        selected_codes = []

    st.divider()

    # Paramètres de génération
    st.subheader("Paramètres de génération")
    temperature   = st.slider("Température (0 = déterministe)", 0.0, 1.0, 0.0, 0.05)
    max_tokens    = st.number_input("Max tokens par réponse", 10, 500, 100, 10)
    delay         = st.slider("Délai entre requêtes (s)", 0.0, 5.0, 1.0, 0.5)
    max_questions = st.number_input("Max questions par langue (0 = illimité)", 0, 10000, 0, 10)


# Métriques rapides

col1, col2, col3 = st.columns(3)
col1.metric("Fichiers output", len(list(OUTPUT_DIR.glob("*.jsonl"))) if OUTPUT_DIR.exists() else 0)
col2.metric("Langues disponibles", len(get_available_languages()))
col3.metric("Provider actif", provider)

st.divider()

# Résumé de la config sélectionnée
if selected_codes:
    model_label = groq_model if provider == "groq" else local_model
    st.info(f"**{provider}/{model_label}** | Langues : {', '.join(selected_codes)} | Dataset : {dataset_type}")

# Lancement du pipeline
launch = st.button(
    "Lancer le run",
    disabled=not selected_codes,
    type="primary",
)

if launch and selected_codes:
    # Construction et sauvegarde de la config
    config = build_config(
        provider, groq_model, local_model, selected_codes, dataset_type,
        temperature, max_tokens, delay, max_questions
    )
    config_path = save_temp_config(config)

    st.subheader("Progression")
    log_area    = st.empty()       # zone de logs mise à jour en temps réel
    progress_bar = st.progress(0)  # barre de progression
    status_text = st.empty()       # texte "X/Y questions"

    # Lancement du pipeline dans un thread séparé
    log_queue = queue.Queue()
    thread = threading.Thread(
        target=run_pipeline_threaded,
        args=(config_path, log_queue),
        daemon=True,
    )
    thread.start()

    # Calcul du nombre total de questions pour la barre de progression
    total_questions = sum(
        count_lines(DATA_DIR / f"{lang}_{dataset_type}.jsonl")
        for lang in selected_codes
    )
    answered = 0
    logs = []

    # Boucle d'affichage : lit les messages du pipeline en temps réel
    while True:
        try:
            msg = log_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if msg == "__DONE__":
            break

        logs.append(msg)
        # Affiche les 30 derniers messages pour ne pas surcharger l'interface
        log_area.code("\n".join(logs[-30:]))

        # Mise à jour de la barre de progression à chaque question réussie
        if "OK —" in msg:
            answered += 1
            if total_questions > 0:
                progress_bar.progress(min(answered / total_questions, 1.0))
            status_text.text(f"{answered}/{total_questions} questions")

    thread.join()
    progress_bar.progress(1.0)
    status_text.text("Tâche complètée")
    st.success("Tâche complètée")
    


# Section export
st.divider()
st.subheader("Export du package de soumission")

output_files = list(OUTPUT_DIR.glob("*.jsonl")) if OUTPUT_DIR.exists() else []

if output_files:
    # Affiche l'état de chaque fichier output (réponses complétées / total)
    for f in sorted(output_files):
        answered = count_answered(f)
        st.write(f"**{f.name}** — {answered} réponses générées")

    # Bouton de téléchargement du zip (JSONL + configs)
    zip_bytes = make_zip()
    st.download_button(
        label="Télécharger le package (.zip)",
        data=zip_bytes,
        file_name=f"eloquent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )
else:
    st.info("Aucun output disponible. Lance un run pour générer des résultats.")
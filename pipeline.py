import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from providers.base import LLMProvider
from providers.api_provider import GroqProvider
from providers.local_provider import LocalProvider

load_dotenv()

# charge la configuration yaml[cite: 2]
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# construit le provider selon la config[cite: 2]
def build_provider(config: dict) -> LLMProvider:
    provider_name = config["provider"]

    if provider_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante dans le fichier .env")
        return GroqProvider(
            api_key=api_key,
            model=config["groq_model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )

    elif provider_name == "local":
        return LocalProvider(
            model=config["local_model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            base_url=config.get("local_base_url", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Provider inconnu : '{provider_name}'")

# configure le logger pour la console et le fichier[cite: 2]
def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# charge les questions deja traitées pour la reprise[cite: 2]
def load_already_done(output_file: Path) -> dict:
    already_done = {}
    if not output_file.exists():
        return already_done

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                if entry.get("answer") is not None:
                    already_done[entry.get("id")] = entry
            except json.JSONDecodeError: pass
    return already_done

# execution d'une langue specifique
def run_language(
    lang: str,
    dataset_type: str,
    provider: LLMProvider,
    config: dict,
    logger: logging.Logger,
    stop_event=None
) -> dict:
    from prompter import Prompter  
    
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialisation du prompter avec le chemin du yaml[cite: 3]
    prompter = Prompter(config_path=config["config_path"])
    variant_name = prompter.get_variant_name()
    
    # nom de fichier unique par variant pour eviter les ecrasements[cite: 1]
    provider_clean = provider.name().replace('/', '_').replace(':', '_')
    output_filename = f"{lang}_{dataset_type}_{provider_clean}_{variant_name}.jsonl"
    output_file = output_dir / output_filename

    input_file = data_dir / f"{lang}_{dataset_type}.jsonl"

    if not input_file.exists():
        logger.warning(f"[{lang}] Fichier source introuvable : {input_file}")
        return {"lang": lang, "total": 0, "success": 0, "errors": 0, "skipped": 0}

    already_done = load_already_done(output_file)
    logger.info(f"[{lang}] Run variant: {variant_name}")

    max_questions = config.get("max_questions", None)
    delay = config.get("delay_seconds", 0)
    total = success = errors = skipped = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        all_entries = [json.loads(l) for l in fin if l.strip()]

    with open(output_file, "w", encoding="utf-8") as fout:
        for line_num, entry in enumerate(all_entries, start=1):
            # gestion de l'arret manuel (bouton stop)[cite: 4]
            if stop_event and stop_event.is_set():
                logger.info(f"[{lang}] Arret demandé par l'utilisateur")
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            entry_id = entry.get("id")
            if entry_id in already_done:
                fout.write(json.dumps(already_done[entry_id], ensure_ascii=False) + "\n")
                skipped += 1
                continue

            if max_questions and (total + 1) > max_questions:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            total += 1
            question_brute = entry.get("question") or entry.get("prompt") or entry.get("text") or ""

            try:
                # application du template depuis le prompter (lot c)[cite: 1]
                final_prompt = prompter.get_prompt(question_brute)
                answer = provider.generate(final_prompt)
                
                entry["answer"] = answer
                entry["model"] = provider.name()
                entry["variant"] = variant_name # tracabilité lot c[cite: 1]
                entry["timestamp"] = datetime.now().isoformat()
                
                success += 1
                logger.info(f"[{lang}] Q{line_num} OK")
            except Exception as e:
                entry["answer"] = None
                entry["variant"] = variant_name
                entry["error"] = str(e)
                errors += 1
                logger.error(f"[{lang}] Q{line_num} ERREUR : {e}")

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if delay > 0: time.sleep(delay)

    return {"lang": lang, "total": total, "success": success, "errors": errors, "skipped": skipped}

def save_submission_metadata(config, timestamp):
    # chemin de sortie dans le dossier outputs
    metadata_path = Path(config["output_dir"]) / f"metadata_{timestamp}.json"
    
    # récupération de la variante et des templates
    variant_active = config.get("variant", "baseline")
    templates = config.get("variants_templates", {})
    
    # extraction du texte du prompt spécifique à la variante (lot c)
    # si c'est la baseline, le système prompt est vide ou neutre
    full_template = templates.get(variant_active, "{question}")

    # construction de l'objet strictement conforme à l'exemple
    metadata = {
        "team": "votre-nom-de-groupe", # à personnaliser
        "system": "eloquent-miage-v1",
        "model": config.get("groq_model") if config["provider"] == "groq" else config.get("local_model"),
        "submissionid": f"experiment-{timestamp}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "label": "eloquent-2026-cultural",
        "languages": config.get("languages", []),
        "modifications": {
            "system_prompt": full_template if variant_active != "baseline" else "Standard LLM prompt",
            "prompt_prefix_generic": full_template.split("{question}")[0] if "{question}" in full_template else "",
            "prompt_suffix_generic": full_template.split("{question}")[1] if "{question}" in full_template else "",
            "generation_params": {
                "temperature": config.get("temperature"),
                "max_tokens": config.get("max_tokens"),
                "do_sample": config.get("temperature") > 0
            },
            "notes": f"Test de la variante {variant_active} sur dataset {config.get('dataset_type')}"
        }
    }

    # écriture du fichier json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# lance le pipeline complet[cite: 2]
def run_pipeline(config_path: str = "config/baseline.yaml", stop_event=None):
    config = load_config(config_path)
    config["config_path"] = config_path 

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logger = setup_logger(str(log_file))
    logger.info(f"=== démarrage pipeline : {config_path} ===")

    provider = build_provider(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    try:
        for lang in config["languages"]:
            if stop_event and stop_event.is_set(): break
            stats = run_language(lang, config["dataset_type"], provider, config, logger, stop_event=stop_event)
            all_stats.append(stats)
    except Exception as e:
        logger.error(f"erreur critique : {e}")
    finally:
        save_submission_metadata(config, timestamp)
        logger.info(f"metadonnées generées dans {config['output_dir']}")
        logger.info(f"log complet : {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
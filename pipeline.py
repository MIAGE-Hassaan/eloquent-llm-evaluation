"""
pipeline.py — Lot A
Lit un fichier JSONL, envoie chaque prompt au LLM configuré,
et écrit les réponses dans un fichier JSONL de sortie.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from providers.base import LLMProvider
from providers.api_provider import GroqProvider
from providers.local_provider import LocalProvider

# Charge les variables d'environnement depuis .env
load_dotenv()


# ---------------------------------------------------------------------------
# Chargement de la configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Construction du provider selon la config
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Provider inconnu : '{provider_name}'. Valeurs acceptées : 'groq', 'local'")


# ---------------------------------------------------------------------------
# Setup du logger
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ---------------------------------------------------------------------------
# Traitement d'un fichier JSONL pour une langue
# ---------------------------------------------------------------------------

def run_language(
    lang: str,
    dataset_type: str,
    provider: LLMProvider,
    config: dict,
    logger: logging.Logger,
) -> dict:
    """
    Traite un fichier JSONL pour une langue donnée.
    Retourne un dict de statistiques du run.
    """
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / f"{lang}_{dataset_type}.jsonl"
    output_file = output_dir / f"{lang}_{dataset_type}_{provider.name().replace('/', '_')}.jsonl"

    if not input_file.exists():
        logger.warning(f"Fichier introuvable, langue ignorée : {input_file}")
        return {"lang": lang, "total": 0, "success": 0, "errors": 0}

    total = success = errors = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            entry = json.loads(line)
            prompt = entry.get("prompt", "")

            try:
                answer = provider.generate(prompt)
                entry["answer"] = answer
                entry["model"] = provider.name()
                success += 1
                logger.info(f"[{lang}] Q{line_num} OK — {answer[:60]}...")

            except Exception as e:
                entry["answer"] = None
                entry["error"] = str(e)
                errors += 1
                logger.error(f"[{lang}] Q{line_num} ERREUR — {e}")

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"[{lang}] Terminé — {success}/{total} réussies, {errors} erreurs → {output_file}")
    return {"lang": lang, "total": total, "success": success, "errors": errors}


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "config/baseline.yaml"):
    config = load_config(config_path)

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logger = setup_logger(str(log_file))
    logger.info(f"=== Démarrage du pipeline — config : {config_path} ===")

    provider = build_provider(config)
    logger.info(f"Provider : {provider.name()}")

    config_copy = Path(config["output_dir"]) / f"config_{timestamp}.yaml"
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, config_copy)
    logger.info(f"Config sauvegardée → {config_copy}")

    dataset_type = config["dataset_type"]
    all_stats = []

    for lang in config["languages"]:
        stats = run_language(lang, dataset_type, provider, config, logger)
        all_stats.append(stats)

    logger.info("=== Résumé ===")
    for s in all_stats:
        logger.info(f"  {s['lang']} : {s['success']}/{s['total']} OK, {s['errors']} erreurs")
    logger.info(f"Log complet : {log_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline ELOQUENT — Lot A")
    parser.add_argument(
        "--config",
        default="config/baseline.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
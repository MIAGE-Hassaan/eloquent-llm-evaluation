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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def setup_logger(log_path: str) -> logging.Logger:
    # Réutilise le logger s'il existe déjà (évite les handlers dupliqués)
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


def load_already_done(output_file: Path) -> dict:
    """
    Lit l'output existant et retourne un dict {id: entry} des questions
    déjà traitées avec succès. Permet de reprendre sans doublons.
    """
    already_done = {}
    if not output_file.exists():
        return already_done

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("answer") is not None:
                    already_done[entry.get("id")] = entry
            except json.JSONDecodeError:
                pass

    return already_done


def find_existing_config(output_dir: Path, provider_name: str) -> Path | None:
    """
    Cherche un fichier de config existant dans output_dir qui correspond
    au même provider. Retourne le chemin s'il existe, sinon None.
    """
    for f in sorted(output_dir.glob("config_*.yaml"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as cfg:
                existing = yaml.safe_load(cfg)
                if existing.get("provider") == provider_name:
                    return f
        except Exception:
            pass
    return None


def run_language(
    lang: str,
    dataset_type: str,
    provider: LLMProvider,
    config: dict,
    logger: logging.Logger,
) -> dict:
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / f"{lang}_{dataset_type}.jsonl"
    output_file = output_dir / f"{lang}_{dataset_type}_{provider.name().replace('/', '_')}.jsonl"

    if not input_file.exists():
        logger.warning(f"Fichier introuvable, langue ignorée : {input_file}")
        return {"lang": lang, "total": 0, "success": 0, "errors": 0, "skipped": 0}

    # Chargement des réponses déjà générées
    already_done = load_already_done(output_file)
    if already_done:
        logger.info(f"[{lang}] Reprise détectée — {len(already_done)} questions déjà traitées dans {output_file.name}")
    else:
        logger.info(f"[{lang}] Nouveau run — {output_file.name}")

    max_questions = config.get("max_questions", None)
    delay = config.get("delay_seconds", 0)

    total = success = errors = skipped = 0

    # Lecture de toutes les entrées du fichier d'entrée
    all_entries = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))

    # Réécriture complète du fichier : déjà traités + nouveaux
    with open(output_file, "w", encoding="utf-8") as fout:

        for line_num, entry in enumerate(all_entries, start=1):
            entry_id = entry.get("id")

            # Question déjà traitée : on recopie telle quelle
            if entry_id in already_done:
                fout.write(json.dumps(already_done[entry_id], ensure_ascii=False) + "\n")
                skipped += 1
                continue

            # Limite de nouvelles questions
            if max_questions and (total + 1) > max_questions:
                # Recopie les entrées restantes sans answer
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            total += 1
            question = entry.get("question") or entry.get("prompt") or entry.get("text") or ""

            if not question:
                logger.warning(f"[{lang}] Q{line_num} — champ question vide ou introuvable")

            try:
                answer = provider.generate(question)
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

            if delay > 0:
                time.sleep(delay)

    logger.info(f"[{lang}] Terminé — {success} nouvelles, {skipped} ignorées, {errors} erreurs → {output_file.name}")
    return {"lang": lang, "total": total, "success": success, "errors": errors, "skipped": skipped}


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

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Réutilise le fichier de config existant si même provider, sinon en crée un nouveau
    existing_config = find_existing_config(output_dir, config["provider"])
    if existing_config:
        logger.info(f"Config existante réutilisée → {existing_config.name}")
    else:
        config_copy = output_dir / f"config_{timestamp}.yaml"
        shutil.copy(config_path, config_copy)
        logger.info(f"Nouvelle config sauvegardée → {config_copy.name}")

    dataset_type = config["dataset_type"]
    all_stats = []

    for lang in config["languages"]:
        stats = run_language(lang, dataset_type, provider, config, logger)
        all_stats.append(stats)

    logger.info("=== Résumé ===")
    for s in all_stats:
        logger.info(f"  {s['lang']} : {s['success']} nouvelles, {s['skipped']} ignorées, {s['errors']} erreurs")
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
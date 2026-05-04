import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from providers.base import LLMProvider
from providers.api_provider import GroqProvider
from providers.local_provider import LocalProvider

load_dotenv()


def clean_answer(answer: str) -> str:
    """
    Post-traite la réponse du LLM pour extraire une réponse propre en une seule phrase.

    Gère les réponses chain-of-thought qui génèrent :
        [raisonnement]… However, here is the answer in exactly one sentence:

        [RÉPONSE FINALE]
    Dans ce cas, seule la dernière partie est conservée.
    """
    if not answer:
        return answer

    # Pattern : "one sentence:\n\nFINAL" ou "one sentence:\nFINAL"
    match = re.search(
        r"(?:one sentence|une phrase|une seule phrase)\s*:\s*\n+(.+?)\s*$",
        answer,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback : si la réponse contient plusieurs paragraphes, garder le dernier
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs[-1]

    return answer.strip()

# Seuls ces champs apparaissent dans les fichiers de soumission ELOQUENT
# L'ordre est important pour la lisibilité et certains scripts d'évaluation
_SUBMISSION_FIELDS = ["id", "prompt", "answer"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_provider(config: dict, system_prompt: str = "") -> LLMProvider:
    provider_name = config["provider"]
    sp = system_prompt if system_prompt else None

    if provider_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante dans le fichier .env")
        return GroqProvider(
            api_key=api_key,
            model=config["groq_model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            system_prompt=sp,
        )

    elif provider_name == "local":
        return LocalProvider(
            model=config["local_model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            base_url=config.get("local_base_url", "http://localhost:11434"),
            system_prompt=sp,
        )

    raise ValueError(f"Provider inconnu : '{provider_name}'")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if logger.handlers:          # déjà configuré (ex. depuis app.py via QueueHandler)
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Résumé résultats
# ---------------------------------------------------------------------------

def _is_valid_answer(answer) -> bool:
    """Retourne False si la réponse est un placeholder simulé ou vide."""
    if not answer:
        return False
    if isinstance(answer, str) and "simul" in answer.lower():
        return False
    return True


def load_already_done(output_file: Path) -> dict:
    """Charge les entrées déjà traitées (réponses valides uniquement) pour la reprise."""
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
                answer = entry.get("answer")
                if _is_valid_answer(answer):
                    already_done[entry["id"]] = entry
            except json.JSONDecodeError:
                pass
    return already_done


# ---------------------------------------------------------------------------
# Export soumission
# ---------------------------------------------------------------------------

def export_submission_file(work_file: Path, submission_dir: Path,
                           lang: str, dataset_type: str) -> Path:
    """
    Génère un JSONL propre (id/prompt/answer uniquement) dans submission_dir,
    au format exact attendu par ELOQUENT : {lang}_{dataset_type}.jsonl.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    out_path = submission_dir / f"{lang}_{dataset_type}.jsonl"

    with open(work_file, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("answer"):
                    # Création d'un dict ordonné (Python 3.7+)
                    clean = {k: entry[k] for k in _SUBMISSION_FIELDS if k in entry}
                    fout.write(json.dumps(clean, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                pass

    return out_path


# ---------------------------------------------------------------------------
# Traitement d'une langue
# ---------------------------------------------------------------------------

def run_language(
    lang: str,
    dataset_type: str,
    provider: LLMProvider,
    config: dict,
    logger: logging.Logger,
    submission_dir: Path = None,
    stop_event=None,
) -> dict:
    from prompter import Prompter

    data_dir      = Path(config["data_dir"])
    output_dir    = Path(config["output_dir"])
    # Dossier de soumission spécifique à l'expérience (passé par run_pipeline ou par défaut)
    if not submission_dir:
        submission_dir = Path("submission") / "latest"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    prompter     = Prompter(config_path=config["config_path"])
    variant_name = prompter.get_variant_name()

    # Fichier de travail dans output_dir racine (compatible avec les fichiers existants)
    provider_clean  = provider.name().replace("/", "_").replace(":", "_")
    work_file = output_dir / f"{lang}_{dataset_type}_{provider_clean}_{variant_name}.jsonl"

    input_file = data_dir / f"{lang}_{dataset_type}.jsonl"
    if not input_file.exists():
        logger.warning(f"[{lang}] Fichier source introuvable : {input_file}")
        return {"lang": lang, "total": 0, "success": 0, "errors": 0, "skipped": 0}

    already_done = load_already_done(work_file)
    logger.info(f"[{lang}] Variant={variant_name} | Reprise={len(already_done)} déjà traités")

    delay         = config.get("delay_seconds", 0)
    max_questions = config.get("max_questions", None)
    total = success = errors = skipped = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        all_entries = [json.loads(l) for l in fin if l.strip()]

    with open(work_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(all_entries):
            # Arrêt manuel
            if stop_event and stop_event.is_set():
                logger.info(f"[{lang}] Arrêt — {success} réponses générées sur {len(all_entries)}")
                # Écrire les entrées restantes sans réponse pour garder la cohérence du fichier
                for remaining in all_entries[idx:]:
                    fout.write(json.dumps(remaining, ensure_ascii=False) + "\n")
                break

            entry_id = entry.get("id")

            # Reprise : déjà traité
            if entry_id in already_done:
                fout.write(json.dumps(already_done[entry_id], ensure_ascii=False) + "\n")
                skipped += 1
                continue

            # Limite optionnelle de questions
            if max_questions and total >= max_questions:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            total += 1
            question_brute = (
                entry.get("prompt") or entry.get("question") or entry.get("text") or ""
            )

            try:
                final_prompt    = prompter.get_prompt(question_brute)
                answer          = provider.generate(final_prompt)
                entry["answer"]    = clean_answer(answer)   # extrait la phrase finale si CoT
                entry["model"]     = provider.name()
                entry["variant"]   = variant_name
                entry["timestamp"] = datetime.now().isoformat()
                success += 1
                logger.info(f"[{lang}] Q{idx + 1}/{len(all_entries)} ✓")
            except Exception as exc:
                entry["answer"]  = None
                entry["variant"] = variant_name
                entry["error"]   = str(exc)
                errors += 1
                logger.error(f"[{lang}] Q{idx + 1} ERREUR : {exc}")

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if delay > 0:
                time.sleep(delay)

    # Export fichier de soumission propre
    if success > 0:
        sub_path = export_submission_file(work_file, submission_dir, lang, dataset_type)
        logger.info(f"[{lang}] → soumission : {sub_path.name} ({success} réponses)")

    return {"lang": lang, "total": total, "success": success,
            "errors": errors, "skipped": skipped}


# ---------------------------------------------------------------------------
# Métadonnées de soumission
# ---------------------------------------------------------------------------

def save_submission_metadata(config: dict, timestamp: str, submission_dir: Path = None) -> Path:
    output_dir = Path(config["output_dir"])
    if not submission_dir:
        submission_dir = Path("submission") / "latest"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegardé dans outputs/ avec timestamp pour historique
    metadata_hist_path = output_dir / f"submission_metadata_{timestamp}.json"
    # Sauvegardé dans submission/ avec le nom fixe requis par la spec
    metadata_final_path = submission_dir / "submission_metadata.json"

    variant_active   = config.get("variant", "baseline")
    templates        = config.get("variants_templates", {})
    system_prompts   = config.get("variants_system_prompts", {})
    full_template    = templates.get(variant_active, "{question}")
    system_prompt    = system_prompts.get(variant_active, "")

    if "{question}" in full_template:
        parts         = full_template.split("{question}", 1)
        prompt_prefix = parts[0].strip()
        prompt_suffix = parts[1].strip()
    else:
        prompt_prefix = full_template.strip()
        prompt_suffix = ""

    model_name = (
        config.get("groq_model", "unknown")
        if config.get("provider") == "groq"
        else config.get("local_model", "unknown")
    )

    metadata = {
        "team":         config.get("team", "votre-nom-de-groupe"),
        "system":       config.get("system", "eloquent-miage-v1"),
        "model":        model_name,
        "submissionid": f"experiment-{timestamp}",
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "label":        config.get("label", "eloquent-2026-cultural"),
        "languages":    config.get("languages", []),
        "modifications": {
            "system_prompt":          system_prompt,
            "prompt_prefix_english":  prompt_prefix,
            "prompt_suffix_english":  prompt_suffix,
            "generation_params": {
                "do_sample":      config.get("temperature", 0) > 0,
                "max_new_tokens": config.get("max_tokens", 200),
            },
            "notes": (
                f"Variante '{variant_active}' | dataset '{config.get('dataset_type')}' "
                f"| provider '{config.get('provider')}'"
            ),
        },
    }

    # Écriture des deux fichiers
    for p in [metadata_hist_path, metadata_final_path]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    return metadata_hist_path


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "config/baseline.yaml", stop_event=None):
    from prompter import Prompter

    config = load_config(config_path)
    config["config_path"] = config_path

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"run_{timestamp}.log"

    logger = setup_logger(str(log_file))
    logger.info(f"=== Démarrage pipeline : {config_path} ===")

    # Récupération du system prompt depuis la variante active
    prompter      = Prompter(config_path)
    system_prompt = prompter.get_system_prompt()
    if system_prompt:
        logger.info(f"System prompt actif pour variante '{prompter.get_variant_name()}'")

    provider = build_provider(config, system_prompt=system_prompt)
    logger.info(f"Provider : {provider.name()}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Création du dossier d'expérience unique
    variant_name = prompter.get_variant_name()
    exp_id = f"{variant_name}_{timestamp}"
    submission_dir = Path("submission") / exp_id
    submission_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    try:
        for lang in config["languages"]:
            if stop_event and stop_event.is_set():
                break
            stats = run_language(
                lang, config["dataset_type"], provider, config, logger,
                submission_dir=submission_dir,
                stop_event=stop_event,
            )
            all_stats.append(stats)
    except Exception as exc:
        logger.error(f"Erreur critique : {exc}")
    finally:
        meta_path = save_submission_metadata(config, timestamp, submission_dir=submission_dir)
        logger.info(f"Métatonnées (soumission) -> {submission_dir / 'submission_metadata.json'}")
        logger.info(f"Log complet  → {log_file}")

    # Résumé
    total_ok = sum(s["success"] for s in all_stats)
    logger.info(f"=== Terminé : {total_ok} réponses générées sur {len(all_stats)} langue(s) ===")
    return all_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline ELOQUENT CLEF 2026")
    parser.add_argument("--config", default="config/baseline.yaml",
                        help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()
    run_pipeline(args.config)
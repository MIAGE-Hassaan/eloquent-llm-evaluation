import yaml


class Prompter:
    def __init__(self, config_path: str = "config/baseline.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.active_variant  = self.config.get("variant", "baseline")
        self.templates       = self.config.get("variants_templates", {})
        self.system_prompts  = self.config.get("variants_system_prompts", {})
        self.current_template = self.templates.get(self.active_variant, "{question}")

    def get_prompt(self, question_text: str) -> str:
        """Injecte la question dans le template de la variante active."""
        return self.current_template.format(question=question_text)

    def get_system_prompt(self) -> str:
        """Retourne le system prompt de la variante active (vide si aucun)."""
        return self.system_prompts.get(self.active_variant, "") or ""

    def get_variant_name(self) -> str:
        """Retourne le nom de la variante pour les métadonnées."""
        return self.active_variant
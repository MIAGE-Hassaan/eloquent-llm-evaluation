import yaml

class Prompter:
    def __init__(self, config_path="baseline.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # On récupère la variante active et son template
        self.active_variant = self.config.get('variant', 'baseline')
        self.templates = self.config.get('variants_templates', {})
        self.current_template = self.templates.get(self.active_variant, "{question}")

    def get_prompt(self, question_text):
        # Injecte la question dans le template de la variante
        return self.current_template.format(question=question_text)

    def get_variant_name(self):
        # Retourne le nom de la variante pour les métadonnées
        return self.active_variant
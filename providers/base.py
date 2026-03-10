from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Classe abstraite que tous les providers doivent implémenter."""

    @abstractmethod
    def generate(self, question: str) -> str:
        """
        Envoie une question au LLM et retourne la réponse sous forme de string.
        Chaque appel est une session indépendante (pas de mémoire entre les questions).
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Retourne le nom du provider/modèle (pour les logs et métadonnées)."""
        pass

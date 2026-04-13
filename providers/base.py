from abc import ABC, abstractmethod

# Classe abstraite que tous les providers doivent implémenter
class LLMProvider(ABC):
    # Envoie une question au LLM et retourne la réponse sous forme de string
    @abstractmethod
    def generate(self, question: str) -> str:
        pass

    # Retourne le nom du provider/modèle (pour les logs et métadonnées)
    @abstractmethod
    def name(self) -> str:
        pass

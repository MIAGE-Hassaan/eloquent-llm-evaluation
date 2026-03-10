from providers.groq_provider import GroqProvider

elif provider_name == "groq":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY manquante dans le fichier .env")
    return GroqProvider(
        api_key=api_key,
        model=config["groq_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
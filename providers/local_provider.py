import asyncio
import json
import requests
import os

#Envoie d'une requête au model local et récupération de la réponse
async def request(model, prompt):
    return requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )

#Ecris une nouvelle ligne dans un fichier json donné
def newLine(id, response, filename):
    data = {}

    #Lecture du contenu existant
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    #Ajout nouvelle ligne
    data[id] = response

    #Réecriture du fichier complet avec la mise à jour
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

async def main():
    model = "gemma3:1b"
    data = None

    #Boucle du fichier specific
    with open("../data/fr_specific.jsonl", "r", encoding="utf-8") as fr_specific:
        for line in fr_specific:
            data = json.loads(line)
            
            response = await request(model, data["prompt"])
            newLine(data["id"], response.json()["response"], "response_fr_specific.json")

            #Retirer pour test réel
            if(data["id"] == "1-10"):
                break
    
    #Boucle du fichier unspecific
    with open("../data/fr_unspecific.jsonl", "r", encoding="utf-8") as fr_unspecific:
        for line in fr_unspecific:
            data = json.loads(line)
            
            response = await request(model, data["prompt"])
            newLine(data["id"], response.json()["response"], "response_fr_unspecific.json")

            #Retirer pour test réel
            if(data["id"] == "10"):
                break

asyncio.run(main())
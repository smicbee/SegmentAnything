import base64
import requests
import json

# Pfad zum Bild, das du testen möchtest
image_path = "test-images/SEM front_002.tif"

# API-Endpunkt
url = "https://segmentanything-app.eu-de-7.icp.infineon.com:5000/segment"

# Punkte und Labels für die Segmentierung (verschiedene Testdaten möglich)
input_data = {
    "points": [[767, 760]],  # Beispielsegmente
    "labels": [1]  # 1 = Positiv, 0 = Negativ
}

# Bild in Base64 umwandeln
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Bild in base64-codierten String umwandeln (Bytes -> String)
        return base64.b64encode(image_file.read()).decode("utf-8")

# Hochladen des Base64-Bilds und Daten als JSON an die API senden
def send_request_to_api(image_path, endpoint, points, labels):
    try:
        # Kodierung des Bildes
        image_base64 = convert_image_to_base64(image_path)
     
        # Erstelle die JSON-Payload
        payload = {
            "image": image_base64,
            "points": points,
            "labels": labels
        }

        # HTTP POST-Anfrage senden
        headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, data=json.dumps(payload), headers=headers)

        # Antwort der API anzeigen
        print("Status Code:", response.status_code)
        print("Antwort JSON:", response.json())  # Zeigt theoretisch auch die Maske
    except Exception as e:
        print(f"Fehler beim Senden der Anfrage: {str(e)}")

# API testen
send_request_to_api(image_path, url, input_data["points"], input_data["labels"])
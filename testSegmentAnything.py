import base64
import requests
import json

# Pfad zum Bild, das du testen möchtest
image_path = "test-images/test.jpg"

# API-Endpunkt
baseUrl = "http://segment-anything-imagelabeling.eu-de-7.icp.infineon.com/"
localUrl = "http://localhost:5000/"
segmentAllEndpoint = "segment_all"

useLocal = True


# Bild in Base64 umwandeln
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Bild in base64-codierten String umwandeln (Bytes -> String)
        return base64.b64encode(image_file.read()).decode("utf-8")

# Hochladen des Base64-Bilds und Daten als JSON an die API senden
def send_request_to_api(image_path, endpoint):
    try:
        # Kodierung des Bildes
        image_base64 = convert_image_to_base64(image_path)
     
        # Erstelle die JSON-Payload
        payload = {
            "image": image_base64
        }

        # HTTP POST-Anfrage senden
        headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, data=json.dumps(payload), headers=headers)

        print(json.dumps(payload))
        # Antwort der API anzeigen
        print("Status Code:", response.status_code)
        print("Antwort JSON:", response.json())  # Zeigt theoretisch auch die Maske
    except Exception as e:
        print(f"Fehler beim Senden der Anfrage: {str(e)}")

# API testen
url = baseUrl + segmentAllEndpoint

if useLocal:
    url = localUrl + segmentAllEndpoint

send_request_to_api(image_path, url)
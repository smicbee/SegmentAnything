from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2  # Für Konturberechnung
from segment_anything import sam_model_registry, SamPredictor
import torch
import base64
from io import BytesIO
import os

# Flask-App initialisieren
app = Flask(__name__)

# Geräte wählen (GPU oder CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM-Modell laden
model_path = "./model/sam_vit_b_01ec64.pth"
print(f"Lade SAM-Modell von: {model_path}")
sam = sam_model_registry["vit_b"](checkpoint=model_path)
sam.to(device=device)
predictor = SamPredictor(sam)

@app.route("/segment", methods=["POST"])
def segment_image():
    """
    REST API für Segment Anything – verarbeitet ein Base64-kodiertes Bild,
    führt Segmentierung durch, und gibt Maske sowie Polygon zurück.
    """
    # JSON-Daten abrufen
    try:
        data = request.get_json()
        image_base64 = data.get("image")  # Base64-kodiertes Bild im JSON
        points = np.array(data.get("points", []))  # Punkte (Koordinaten)
        labels = np.array(data.get("labels", []))  # Labels (1 = positiv, 0 = negativ)

        # Validierung von Base64-Bild und Punkt-/Label-Eingaben
        if not image_base64:
            return jsonify({"error": "Kein Bild im Request vorhanden"}), 400
        if len(points) != len(labels):
            return jsonify({"error": "Ungültige Punkte- oder Label-Daten"}), 400
    except Exception as e:
        return jsonify({"error": f"Fehler beim Verarbeiten der JSON-Daten: {str(e)}"}), 400

    # Base64-Bild zu Image dekodieren
    try:
        image_data = base64.b64decode(image_base64)  # Dekodiere das Base64-Bild
        image = Image.open(BytesIO(image_data)).convert("RGB")  # Konvertiere Bild in RGB
        image_np = np.asarray(image, dtype=np.uint8)  # Konvertiere in uint8 (Bereich: 0-255)
    except Exception as e:
        return jsonify({"error": f"Fehler beim Dekodieren des Bildes: {str(e)}"}), 400

    # Bild in SAM übergeben
    try:
        predictor.set_image(image_np)  # Übergabe der Maske
    except Exception as e:
        return jsonify({"error": f"Fehler beim Setzen des Bildes: {str(e)}"}), 500

    predictor.set_image(image_np)

    # Segmentierung durchführen
    try:
        masks, scores, _ = predictor.predict(
            point_coords=points,         # Punkt-Koordinaten
            point_labels=labels,         # Labels (1 = positiv, 0 = negativ)
            multimask_output=False       # Nur eine maske
        )
    except Exception as e:
        return jsonify({"error": f"Fehler bei der Segmentierung: {str(e)}"}), 500

    # Maske erstellen und als Binärmaske (NumPy -> Base64) zurückgeben
    try:
        mask_data = (masks[0] * 255).astype(np.uint8)  # Binary-Maske (0 = Hintergrund, 255 = Vordergrund)
        mask_image = Image.fromarray(mask_data)  # Maske in ein Bild umwandeln
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")  # Als PNG im Speicher speichern
        mask_base64 = base64.b64encode(buffer.getvalue()).decode()  # Base64-kodierter Maskenstring
    except Exception as e:
        return jsonify({"error": f"Fehler beim Erstellen der Maske: {str(e)}"}), 500

    # Polygon um die Maske herum berechnen
    try:
        contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Konturen finden
        largest_contour = max(contours, key=cv2.contourArea)  # Größte Kontur auswählen
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # Proportionaler Toleranzwert
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)  # Polygon vereinfachen
        polygon_points = approx_polygon[:, 0, :].tolist()  # Polygonpunkte extrahieren (x, y)
    except Exception as e:
        return jsonify({"error": f"Fehler beim Berechnen des Polygons: {str(e)}"}), 500

    # Erfolgsnachricht mit Maske, Polygon und Modellvertrauen
    return jsonify({
        "status": "success",
        "score": float(scores[0]),  # Vertrauen des Modells
        "mask": mask_base64,        # Base64-kodiertes Maskenbild
        "polygon": polygon_points  # Polygonpunkte als (x, y)-Koordinaten
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
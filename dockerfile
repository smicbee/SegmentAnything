FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Arbeitsverzeichnis setzen
WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean

# Projektdateien kopieren
COPY . .
COPY ./model /app/model

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model
RUN wget -O /app/model/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


# REST API starten
EXPOSE 5000
CMD ["python", "app.py"]
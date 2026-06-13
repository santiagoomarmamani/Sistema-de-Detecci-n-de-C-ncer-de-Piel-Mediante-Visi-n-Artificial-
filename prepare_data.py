import pandas as pd
import os

# Clases del dataset
CLASSES = {
    "mel": 0,      # Melanoma
    "nv": 1,       # Nevo melanocitico
    "bcc": 2,      # Carcinoma basocelular
    "akiec": 3,    # Queratosis actinica
    "bkl": 4,      # Queratosis benigna
    "df": 5,       # Dermatofibroma
    "vasc": 6      # Lesion vascular
}

# Leer metadata
df = pd.read_csv("data/HAM10000_metadata.csv")

# Construir rutas de imágenes
def find_image_path(image_id):
    for folder in ["data/images"]:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

df["image_path"] = df["image_id"].apply(find_image_path)
df["label"] = df["dx"].map(CLASSES)

# Eliminar filas sin imagen
df = df.dropna(subset=["image_path", "label"])

# Guardar
df[["image_path", "label"]].to_csv("data/labels.csv", index=False)
print(f"Listo. {len(df)} imágenes encontradas.")
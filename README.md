#Step 1: Installing the required packages:

!pip install -q transformers accelerate timm torch torchvision torchaudio opencv-python pandas

#Step 2: Uploading the necessary zip file:

from google.colab import files
import zipfile, os

uploaded = files.upload() 

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("prescriptions")

print("Extracted to /content/prescriptions")

#Step 3: Loading the BLIP-2 FLAN Model:

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded on", device)

#Step 4: Define Image-to-Text Function

from PIL import Image

def extract_text_from_image(img_path):
    image = Image.open(img_path).convert('RGB')
    prompt = "What medicines, dosages and frequencies are mentioned in this prescription?"
    inputs = processor(image, prompt, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=150)
    return processor.decode(out[0], skip_special_tokens=True)

#Step 5: Running on all the prescription images in the dataset:

results = []

for root, _, files in os.walk("prescriptions"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            try:
                text = extract_text_from_image(path)
                results.append({"filename": file, "extracted_text": text})
                print(f" Processed: {file}")
            except Exception as e:
                print(f" Error on {file}: {e}")

#Step 6: Structuring Output into a DataFrame:

import re
import pandas as pd

def structure_prescription(text):
    meds = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    dosage = re.findall(r'\d+\s?(mg|ML|ml)', text)
    freq = re.findall(r'\d+x\d+|\d+ times a day|\d+ per day', text)
    return {
        "medicines": meds,
        "dosage": dosage,
        "frequency": freq
    }

structured_data = []
for r in results:
    structured = structure_prescription(r["extracted_text"])
    structured["archive.zip"] = r["archive.zip"]
    structured_data.append(structured)

df = pd.DataFrame(structured_data)
df.to_csv("structured_prescriptions.csv", index=False)
df.head()

#Step 7: Our evaluation strategy: The Exact Match Score

manual_labels = [
    {'filename': 'rx1.jpg', 'medicines': ['Paracetamol'], 'dosage': ['500mg'], 'frequency': ['1x2']},
    {'filename': 'rx2.jpg', 'medicines': ['Amoxicillin'], 'dosage': ['250mg'], 'frequency': ['2x1']},
    # Add more as needed
]

def exact_match(predicted, actual):
    return int(predicted == actual)

score = 0
total = len(manual_labels)

for label in manual_labels:
    pred_row = df[df['filename'] == label['filename']]
    if not pred_row.empty:
        pred = pred_row.iloc[0]
        score += exact_match(pred['medicines'], label['medicines'])

accuracy = score / total
print(f"Exact Match Accuracy: {accuracy:.2f}")

#Step 8: Saving to Drive:

df.to_csv("/content/drive/MyDrive/structured_prescriptions.csv", index=False)


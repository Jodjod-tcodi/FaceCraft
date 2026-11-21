"""
generate_sample_data.py - Helper to generate sample face images and metadata.

Creates sample test images and metadata so that the preprocessing pipeline,
DataLoaders, and baseline model can be validated and run immediately.
"""

import os
import random
import pandas as pd
from PIL import Image, ImageDraw
from text_processing import CELEBA_ATTRIBUTES, attributes_to_caption


def create_synthetic_face_sample(filepath: str, width: int = 178, height: int = 218, attributes: dict = None):
    """
    Generates a stylized synthetic face image imitating CelebA raw dimensions (178x218).
    """
    img = Image.new("RGB", (width, height), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)

    # Background gradient or solid
    bg_color = (random.randint(180, 240), random.randint(180, 240), random.randint(200, 250))
    draw.rectangle([0, 0, width, height], fill=bg_color)

    # Face oval
    is_pale = attributes.get("Pale_Skin", -1) == 1 if attributes else False
    skin_tone = (255, 230, 210) if is_pale else (
        random.choice([(240, 200, 160), (220, 175, 140), (190, 140, 100), (130, 90, 60)])
    )
    face_box = [30, 40, width - 30, height - 35]
    draw.ellipse(face_box, fill=skin_tone)

    # Hair
    hair_color = (40, 30, 20)
    if attributes:
        if attributes.get("Blond_Hair", -1) == 1:
            hair_color = (220, 190, 90)
        elif attributes.get("Brown_Hair", -1) == 1:
            hair_color = (90, 55, 30)
        elif attributes.get("Gray_Hair", -1) == 1:
            hair_color = (180, 180, 180)

    if not (attributes and attributes.get("Bald", -1) == 1):
        draw.chord([25, 20, width - 25, 120], 180, 360, fill=hair_color)

    # Eyes
    eye_y = 95
    eye_spacing = 30
    center_x = width // 2
    draw.ellipse([center_x - eye_spacing - 8, eye_y, center_x - eye_spacing + 8, eye_y + 10], fill=(255, 255, 255))
    draw.ellipse([center_x + eye_spacing - 8, eye_y, center_x + eye_spacing + 8, eye_y + 10], fill=(255, 255, 255))
    draw.ellipse([center_x - eye_spacing - 3, eye_y + 2, center_x - eye_spacing + 3, eye_y + 8], fill=(30, 20, 10))
    draw.ellipse([center_x + eye_spacing - 3, eye_y + 2, center_x + eye_spacing + 3, eye_y + 8], fill=(30, 20, 10))

    # Eyeglasses
    if attributes and attributes.get("Eyeglasses", -1) == 1:
        draw.rectangle([center_x - eye_spacing - 12, eye_y - 4, center_x - eye_spacing + 12, eye_y + 14], outline=(20, 20, 20), width=2)
        draw.rectangle([center_x + eye_spacing - 12, eye_y - 4, center_x + eye_spacing + 12, eye_y + 14], outline=(20, 20, 20), width=2)
        draw.line([center_x - eye_spacing + 12, eye_y + 5, center_x + eye_spacing - 12, eye_y + 5], fill=(20, 20, 20), width=2)

    # Nose
    draw.line([center_x, 105, center_x - 3, 125, center_x + 3, 125], fill=(160, 110, 80), width=2)

    # Mouth / Smile
    is_smiling = attributes.get("Smiling", -1) == 1 if attributes else False
    if is_smiling:
        draw.arc([center_x - 20, 135, center_x + 20, 160], 0, 180, fill=(180, 50, 50), width=3)
    else:
        draw.line([center_x - 15, 145, center_x + 15, 145], fill=(180, 50, 50), width=2)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath)


def generate_sample_dataset(output_dir: str = "data/samples", num_samples: int = 60):
    """
    Generates a full mini-dataset with images, captions, and attributes.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records = []
    random.seed(42)

    for i in range(1, num_samples + 1):
        img_id = f"{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_id)

        # Generate realistic random attributes
        attr = {a: -1 for a in CELEBA_ATTRIBUTES}
        attr["Male"] = 1 if random.random() > 0.5 else -1
        attr["Young"] = 1 if random.random() > 0.25 else -1
        attr["Smiling"] = 1 if random.random() > 0.45 else -1
        if random.random() > 0.7:
            attr["Eyeglasses"] = 1

        hair = random.choice(["Blond_Hair", "Brown_Hair", "Black_Hair", "Gray_Hair", "Bald"])
        attr[hair] = 1

        caption = attributes_to_caption(attr, seed=i)
        create_synthetic_face_sample(img_path, attributes=attr)

        # Split: 70% train, 15% val, 15% test
        if i <= int(num_samples * 0.7):
            split = 0
        elif i <= int(num_samples * 0.85):
            split = 1
        else:
            split = 2

        record = {
            "image_id": img_id,
            "caption": caption,
            "split": split,
            "source": "celeba_attr"
        }
        record.update(attr)
        records.append(record)

    df = pd.DataFrame(records)
    metadata_path = os.path.join(output_dir, "sample_metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Generated {num_samples} sample images in {images_dir}")
    print(f"Saved metadata to {metadata_path}")
    return df


if __name__ == "__main__":
    generate_sample_dataset()


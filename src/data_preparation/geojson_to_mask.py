"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    geojson_to_mask.py

Description:
    Conversion of QuPath GeoJSON annotations into
    segmentation masks compatible with the AI pipeline.

Author:
    Hamza Graïn
"""

import json
import cv2
import numpy as np
import openslide


#### CONFIGURATION


NDPI_PATH = r"path/to/your/slide.ndpi"

GEOJSON_PATH = r"path/to/your/annotations.geojson"

OUTPUT_MASK = r"path/to/your/output_mask.png"

LEVEL = 4


##### CLASSES


CLASS_MAP = {

    "Vide": 0,

    "CK14": 1,

    "Stroma": 2,

    "Autres": 3

}

CLASS_NAMES = list(CLASS_MAP.keys())


####### OUVERTURE WSI


print("=" * 60)
print("Ouverture de la lame")
print("=" * 60)

slide = openslide.OpenSlide(
    NDPI_PATH
)

base_width, base_height = slide.level_dimensions[0]

width, height = slide.level_dimensions[LEVEL]

downsample = slide.level_downsamples[LEVEL]

scale_x = width / base_width
scale_y = height / base_height

print()

print(f"Niveau 0 : {base_width} x {base_height}")

print(f"Niveau {LEVEL} : {width} x {height}")

print(f"Downsample : {downsample}")

print()

#### MASQUES 

print("=" * 60)
print("Création des masques")
print("=" * 60)

masks = {}

for cls in CLASS_NAMES:

    masks[cls] = np.zeros(

        (height, width),

        dtype=np.uint8

    )

print(f"{len(masks)} masques créés")

print()

##### LECTURE DU GEOJSON

print("=" * 60)
print("Lecture du GeoJSON")
print("=" * 60)

with open(
    GEOJSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    geojson = json.load(f)

print()

print(f"Nombre d'annotations : {len(geojson['features'])}")

print()

#### REGROUPEMENT PAR CLASSE

features_by_class = {

    cls: []

    for cls in CLASS_NAMES

}

for feature in geojson["features"]:

    props = feature.get("properties", {})

    classification = props.get("classification")

    if classification is None:
        continue

    class_name = classification["name"]

    if class_name not in features_by_class:
        continue

    features_by_class[class_name].append(feature)

print("=" * 60)
print("Répartition")
print("=" * 60)

for cls in CLASS_NAMES:

    print(

        f"{cls:<10} : {len(features_by_class[cls])}"

    )

print()


#### CONVERSION D'UN RING


def ring_to_points(ring):

    pts = []

    for x, y in ring:

        xx = int(round(x * scale_x))
        yy = int(round(y * scale_y))

        xx = np.clip(
            xx,
            0,
            width - 1
        )

        yy = np.clip(
            yy,
            0,
            height - 1
        )

        pts.append(

            [xx, yy]

        )

    pts = np.asarray(

        pts,

        dtype=np.int32

    )

    if len(pts) < 3:
        return None

    return pts


#### DESSIN D'UN POLYGONE


def draw_polygon(

    mask,

    polygon

):

    if len(polygon) == 0:
        return

    outer = ring_to_points(

        polygon[0]

    )

    if outer is None:
        return

    cv2.fillPoly(

        mask,

        [outer],

        255

    )

    # trous

    for hole in polygon[1:]:

        hole = ring_to_points(

            hole

        )

        if hole is None:
            continue

        cv2.fillPoly(

            mask,

            [hole],

            0

        )

print("=" * 60)
print("Initialisation terminée")
print("=" * 60)
print()


#### RASTERISATION DES CLASSES


print("=" * 60)
print("Rasterisation")
print("=" * 60)

annotation_count = 0

for class_name in CLASS_NAMES:

    class_mask = masks[class_name]

    features = features_by_class[class_name]

    print()

    print(f"Classe : {class_name}")

    print(f"Annotations : {len(features)}")

    for feature in features:

        geometry = feature.get("geometry")

        if geometry is None:
            continue

        geom_type = geometry["type"]

        
        # POLYGON
        

        if geom_type == "Polygon":

            draw_polygon(

                class_mask,

                geometry["coordinates"]

            )

            annotation_count += 1

       
        # MULTIPOLYGON
       

        elif geom_type == "MultiPolygon":

            for polygon in geometry["coordinates"]:

                draw_polygon(

                    class_mask,

                    polygon

                )

            annotation_count += 1

print()

print("=" * 60)
print("Rasterisation terminée")
print("=" * 60)

print()

print(f"Nombre total d'annotations : {annotation_count}")

print()


#### SAUVEGARDE DES MASQUES INDIVIDUELS


print("=" * 60)
print("Sauvegarde des masques")
print("=" * 60)

for class_name in CLASS_NAMES:

    filename = f"mask_{class_name.lower()}.png"

    cv2.imwrite(

        filename,

        masks[class_name]

    )

    pixels = np.count_nonzero(

        masks[class_name]

    )

    print(

        f"{filename:<20} {pixels} pixels"

    )

print()


#### ANALYSE DES CONFLITS


print("=" * 60)
print("Analyse des conflits")
print("=" * 60)

conflicts = {}

pairs = [

    ("CK14", "Stroma"),

    ("CK14", "Autres"),

    ("Stroma", "Autres"),

    ("Vide", "CK14"),

    ("Vide", "Stroma"),

    ("Vide", "Autres")

]

for c1, c2 in pairs:

    conflict = cv2.bitwise_and(

        masks[c1],

        masks[c2]

    )

    conflicts[(c1, c2)] = conflict

    pixels = np.count_nonzero(conflict)

    print(f"{c1:<8} ∩ {c2:<8} : {pixels} pixels")

    filename = f"conflict_{c1.lower()}_{c2.lower()}.png"

    cv2.imwrite(

        filename,

        conflict

    )

print()


#### FUSION DES MASQUES


print("=" * 60)
print("Fusion des masques")
print("=" * 60)

mask = np.zeros(

    (height, width),

    dtype=np.uint8

)


# PRIORITE

#
# Le principe est simple :
#
# chaque masque est indépendant.
#
# On ne dessine jamais directement dans le masque final.
#
# La priorité est appliquée UNE SEULE FOIS.


priority = [

    ("Vide",0),

    ("Stroma",2),

    ("CK14",1),

    ("Autres",3)

]

for class_name, class_id in priority:

    binary = masks[class_name] > 0

    mask[binary] = class_id

print()

print("Fusion terminée")

print()


#### DISTRIBUTION


print("=" * 60)
print("Distribution")
print("=" * 60)

total = np.count_nonzero(mask)

for class_name, class_id in CLASS_MAP.items():

    pixels = np.count_nonzero(

        mask == class_id

    )

    percent = 100 * pixels / total if total else 0

    print(

        f"{class_name:<10} "

        f"{pixels:>10} "

        f"({percent:5.2f}%)"

    )

print()


#### SAUVEGARDE


cv2.imwrite(

    OUTPUT_MASK,

    mask

)

print("=" * 60)
print("Masque final sauvegardé")
print("=" * 60)

print(OUTPUT_MASK)

print()


#### COULEURS


COLORS = {

    0: (0, 0, 0),

    1: (0, 0, 255),          # CK14 (bleu)

    2: (255, 105, 180),      # Stroma (rose)

    3: (139, 69, 19)         # Autres (marron)

}


#### CREATION D'UN OVERLAY


def create_overlay(mask):

    overlay = np.zeros(

        (height, width, 3),

        dtype=np.uint8

    )

    for cls, color in COLORS.items():

        overlay[mask == cls] = color

    return overlay


#### OVERLAY FINAL


overlay = create_overlay(mask)

cv2.imwrite(

    "mask_overlay.png",

    cv2.cvtColor(

        overlay,

        cv2.COLOR_RGB2BGR

    )

)

#### OVERLAY DES MASQUES INDIVIDUELS


for class_name, class_id in CLASS_MAP.items():

    binary = np.zeros(

        (height, width),

        dtype=np.uint8

    )

    binary[masks[class_name] > 0] = class_id

    overlay = create_overlay(binary)

    cv2.imwrite(

        f"overlay_{class_name.lower()}.png",

        cv2.cvtColor(

            overlay,

            cv2.COLOR_RGB2BGR

        )

    )


#### VISUALISATION DES CONFLITS


print("=" * 60)
print("Création des cartes de conflits")
print("=" * 60)

for (c1, c2), conflict in conflicts.items():

    rgb = np.zeros(

        (height, width, 3),

        dtype=np.uint8

    )

    rgb[conflict > 0] = (255, 255, 255)

    cv2.imwrite(

        f"conflict_{c1.lower()}_{c2.lower()}_overlay.png",

        cv2.cvtColor(

            rgb,

            cv2.COLOR_RGB2BGR

        )

    )

print()


#### QA REPORT


print("=" * 60)
print("QUALITY REPORT")
print("=" * 60)

print()

print("Nombre d'annotations :")

for cls in CLASS_NAMES:

    print(

        f"{cls:<10} "

        f"{len(features_by_class[cls])}"

    )

print()

print("Pixels par classe :")

for cls in CLASS_NAMES:

    pixels = np.count_nonzero(

        masks[cls]

    )

    print(

        f"{cls:<10} "

        f"{pixels}"

    )

print()

print("Pixels du masque final :")

for cls, cid in CLASS_MAP.items():

    pixels = np.count_nonzero(

        mask == cid

    )

    print(

        f"{cls:<10} "

        f"{pixels}"

    )

print()

print("Conflits :")

for (c1, c2), conflict in conflicts.items():

    pixels = np.count_nonzero(

        conflict

    )

    print(

        f"{c1:<8} / {c2:<8} "

        f"{pixels}"

    )

print()


#### EXPORT QA


with open(

    "qa_report.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write("=====================================\n")

    f.write("GeoJSON to Mask V3\n")

    f.write("=====================================\n\n")

    f.write("Annotations\n")

    for cls in CLASS_NAMES:

        f.write(

            f"{cls} : "

            f"{len(features_by_class[cls])}\n"

        )

    f.write("\n")

    f.write("Pixels par classe\n")

    for cls in CLASS_NAMES:

        pixels = np.count_nonzero(

            masks[cls]

        )

        f.write(

            f"{cls} : "

            f"{pixels}\n"

        )

    f.write("\n")

    f.write("Conflits\n")

    for (c1, c2), conflict in conflicts.items():

        pixels = np.count_nonzero(

            conflict

        )

        f.write(

            f"{c1} / {c2} : "

            f"{pixels}\n"

        )

print()

print("=" * 60)
print("Pipeline terminé")
print("=" * 60)

print()

print("Fichiers générés :")

print()

print("- mask_overlay.png")

print("- overlay_ck14.png")

print("- overlay_stroma.png")

print("- overlay_autres.png")

print("- overlay_vide.png")

print()

print("- conflict_ck14_stroma_overlay.png")

print("- conflict_ck14_autres_overlay.png")

print("- conflict_stroma_autres_overlay.png")

print("- conflict_vide_ck14_overlay.png")

print("- conflict_vide_stroma_overlay.png")

print("- conflict_vide_autres_overlay.png")

print()

print("- qa_report.txt")

print()

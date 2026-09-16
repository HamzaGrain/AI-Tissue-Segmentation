"""
P16-KLHL35 AI Annotation Pipeline
---------------------------------

Module:
    geojson_export.py

Description:
    Export of predicted tissue compartment geometries
    to GeoJSON format compatible with QuPath.

    The module converts Shapely geometries into GeoJSON
    features, assigns tissue classifications and colors,
    and scales the coordinates to the original slide level.

Author:
    Hamza Graïn
"""

import json
import os

from shapely.geometry import mapping

from config import *



# SHAPELY -> FEATURE


def create_feature(annotation):

    poly = annotation["polygon"]

    class_id = annotation["class_id"]

    geometry = mapping(poly)

    feature = {

        "type": "Feature",

        "geometry": geometry,

        "properties": {

            "objectType": "annotation",

            "classification": {

                "name": CLASS_NAMES[class_id],

                "color": list(CLASS_COLORS[class_id])

            }

        }

    }

    return feature



# MISE A L'ECHELLE


def scale_geometry(geometry, downsample):

    geom = geometry.copy()

    if geom["type"] == "Polygon":

        rings = []

        for ring in geom["coordinates"]:

            new_ring = []

            for x, y in ring:

                new_ring.append([

                    float(x * downsample),

                    float(y * downsample)

                ])

            rings.append(new_ring)

        geom["coordinates"] = rings

    elif geom["type"] == "MultiPolygon":

        polygons = []

        for poly in geom["coordinates"]:

            rings = []

            for ring in poly:

                new_ring = []

                for x, y in ring:

                    new_ring.append([

                        float(x * downsample),

                        float(y * downsample)

                    ])

                rings.append(new_ring)

            polygons.append(rings)

        geom["coordinates"] = polygons

    return geom



# EXPORT


def export_geojson(annotations, slide_info):

    print("=" * 60)
    print("Export GeoJSON")
    print("=" * 60)

    downsample = slide_info["downsample"]

    features = []

    for annotation in annotations:

        feature = create_feature(annotation)

        feature["geometry"] = scale_geometry(

            feature["geometry"],

            downsample

        )

        features.append(feature)

    geojson = {

        "type": "FeatureCollection",

        "features": features

    }

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True

    )

    output_path = os.path.join(

        OUTPUT_DIR,

        GEOJSON_NAME

    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            geojson,

            f,

            indent=2

        )

    print()
    print(f"{len(features)} annotations exportées.")
    print(output_path)
    print()

    return output_path

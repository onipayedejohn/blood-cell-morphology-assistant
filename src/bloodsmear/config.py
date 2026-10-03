"""Class definitions, morphology notes and paths shared by training and the app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# The environment variables let a preview bundle be tested without touching models/
MODEL_DIR = Path(os.environ.get("BLOODSMEAR_MODEL_DIR", ROOT / "models"))
REPORT_DIR = ROOT / "reports"
METRICS_DIR = Path(os.environ.get("BLOODSMEAR_METRICS_DIR", REPORT_DIR / "metrics"))

# Label ids follow the source annotation files (0-6). Platelets are not in the
# source dataset used here, so the model has seven classes.
CLASSES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
]

# Filename prefix in the PBC dataset -> (class id, finer subtype label)
PREFIX_TO_SUBTYPE = {
    "BA": "Basophil",
    "EO": "Eosinophil",
    "ERB": "Erythroblast",
    "PMY": "Promyelocyte",
    "MY": "Myelocyte",
    "MMY": "Metamyelocyte",
    "IG": "Immature granulocyte (unspecified)",
    "LY": "Lymphocyte",
    "MO": "Monocyte",
    "SNE": "Segmented neutrophil",
    "BNE": "Band neutrophil",
    "NEUTROPHIL": "Neutrophil (unspecified)",
}

# White cells counted in a leukocyte differential. Erythroblasts are nucleated
# red cells and are reported per 100 white cells, not as part of the 100%.
LEUKOCYTES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "immature_granulocyte"]


@dataclass(frozen=True)
class CellInfo:
    name: str
    looks_like: str
    clinical_note: str
    adult_range: str | None  # typical adult % of leukocytes, None if not expected


CELL_INFO = {
    "basophil": CellInfo(
        "Basophil",
        "Coarse, dark purple-black granules that often cover the nucleus.",
        "The rarest white cell in health. Raised counts are seen in myeloproliferative "
        "neoplasms such as chronic myeloid leukemia, and in some allergic states.",
        "0-1%",
    ),
    "eosinophil": CellInfo(
        "Eosinophil",
        "Usually a two-lobed nucleus with large, even, orange-red granules.",
        "Raised in allergy, asthma and parasitic infection, including the helminth "
        "infections common in many tropical settings.",
        "1-6%",
    ),
    "erythroblast": CellInfo(
        "Erythroblast (nucleated red cell)",
        "A round nucleus that becomes smaller and denser as the cell matures, with cytoplasm "
        "that shifts from blue to pink.",
        "Not expected in adult peripheral blood. Seen with hemolysis, for example a sickle "
        "cell crisis, severe anemia, marrow infiltration and in newborns. Reported per 100 "
        "white cells. If the analyzer does not exclude them, about 5 or more per 100 white "
        "cells usually means the white cell count needs correcting.",
        None,
    ),
    "immature_granulocyte": CellInfo(
        "Immature granulocyte",
        "Promyelocytes, myelocytes and metamyelocytes: larger cells with a round to "
        "kidney-shaped, unsegmented nucleus and granular cytoplasm.",
        "Usually absent on a manual differential, though analyzers often report up to about 0.5-1% "
        "in healthy adults. More than that points to a left shift: severe infection or sepsis, "
        "pregnancy, G-CSF treatment, or a myeloproliferative disorder.",
        "0-1%",
    ),
    "lymphocyte": CellInfo(
        "Lymphocyte",
        "A small round cell with a dense, round nucleus and a thin rim of pale blue cytoplasm.",
        "Raised in viral infections and chronic lymphocytic leukemia. Reactive lymphocytes "
        "are larger, with more abundant blue cytoplasm.",
        "20-45%",
    ),
    "monocyte": CellInfo(
        "Monocyte",
        "The largest normal white cell: a folded or kidney-shaped nucleus, lacy chromatin, "
        "gray-blue cytoplasm and sometimes vacuoles.",
        "Raised in chronic infections such as tuberculosis, recovery from marrow suppression, "
        "and chronic myelomonocytic leukemia.",
        "2-10%",
    ),
    "neutrophil": CellInfo(
        "Neutrophil",
        "Segmented forms have 2-5 lobes joined by thin strands; band forms have a curved, "
        "unsegmented nucleus. Pale pink cytoplasm with fine lilac granules.",
        "Raised in bacterial infection, inflammation and stress. Low counts follow chemotherapy, "
        "some drugs and some viral infections.",
        "40-75%",
    ),
}

# Parsed numeric ranges for the differential flags. These are typical adult ranges
# from mostly Western reference populations; laboratories should use their own.
REFERENCE_RANGES = {
    "neutrophil": (40, 75),
    "lymphocyte": (20, 45),
    "monocyte": (2, 10),
    "eosinophil": (1, 6),
    "basophil": (0, 1),
    "immature_granulocyte": (0, 1),
}


def display_name(cls: str) -> str:
    return CELL_INFO[cls].name

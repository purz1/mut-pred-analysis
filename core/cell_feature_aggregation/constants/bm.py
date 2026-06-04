from pathlib import Path
CELL_MODEL_PATH = f"{Path(__file__).resolve().parent.parent}/cell_quality_models"

CELL_LABEL_MAP = {
    0: "Artefacts",
    1: "Basophils",
    2: "Blasts",
    3: "Eosinophils_immature",
    4: "Eosinophils",
    5: "Erythroblasts",
    6: "Lymphocytes",
    7: "Macrophages",
    8: "Megakaryocytes",
    9: "Metamyelocytes",
    10: "Monocytes",
    11: "Myelocytes",
    12: "Neutrophils",
    13: "Plasma_cells",
    14: "Proerythroblasts",
    15: "Promonocytes",
    16: "Promyelocytes",
    -1: "Unknowns",  # This is an extra class not present in the cell classification model
}

REV_CELL_LABEL_MAP = {v: k for k, v in CELL_LABEL_MAP.items()}

CELL_STATISTICS_CLASSES = list(range(1, 17))

ERYTHROBLAST_QUALITY_CLASSES = [
    5, # Erythroblasts
]

GRANULOCYTE_QUALITY_CLASSES = [
    1, # Basophils
    3, # Eosinophils_immature
    4, # Eosinophils
    9, # Metamyelocytes
    11, # Myelocytes
    12, # Neutrophils
]

INCLUDE_LOW_QUALITY = [
    8, # Megakaryocytes
    7, # Macrophages
]

QUALITY_FILTER_STATISTICS = [
    "low_quality",
    "cytoplasm_area",
    "nucleus_area",
    "nucleus_compactness",
    "nucleus_roundness",
    "entire_area",
    "entire_compactness",
    "entire_roundness",
]

CELL_TYPES_TO_FILTER = [
    1, # Basophils
    2, # Blasts
    4, # Eosinophils
    3, # Eosinophils_immature
    5, # Erythroblasts
    6, # Lymphocytes
    9, # Metamyelocytes
    10, # Monocytes
    11, # Myelocytes
    12, # Neutrophils
    13, # Plasma_cells
    14, # Proerythroblasts
    15, # Promonocytes
    16, # Promyelocytes
]

CELL_LINEAGES = {
    "Living_cells": [
        1, # Basophils
        2, # Blasts
        3, # Eosinophils_immature
        4, # Eosinophils
        5, # Erythroblasts
        6, # Lymphocytes
        7, # Macrophages
        8, # Megakaryocytes
        9, # Metamyelocytes
        10, # Monocytes
        11, # Myelocytes
        12, # Neutrophils
        13, # Plasma_cells
        14, # Proerythroblasts
        15, # Promonocytes
        16, # Promyelocytes
    ],
    "Granulopoetic_cells": [
        1, # Basophils
        3, # Eosinophils_immature
        4, # Eosinophils
        9, # Metamyelocytes
        11, # Myelocytes
        12, # Neutrophils
    ],
    "Monocytic_cells": [
        7, # Macrophages
        10, # Monocytes
        15, # Promonocytes
    ],
    "Erythropoetic_cells": [
        5, # Erythroblasts
        14, # Proerythroblasts
    ],
    "Lymphoid_cells": [
        6, # Lymphocytes
        13, # Plasma_cells
    ],
    "Granulocytes": [
        1, # Basophils
        4, # Eosinophils
        12, # Neutrophils
    ],
    "APL": [
        2, # Blasts
        11, # Myelocytes
        16, # Promyelocytes
    ]
}

# Used during the filtering process
STAT_MISC_COLUMNS = [
    "class_label",
    "x0",
    "y0",
    "x1",
    "y1",
    "im_width",
    "im_height",
    "low_quality",
    "mgk_broken",
    "mgk_unclear",
    "mgk_sufficient",
    "image_name",
    "nucleus_roundness", # Required for the filtering step when robust=True
    "entire_roundness", # Required for the filtering step when robust=True
]

# Used during the filtering process
RES_MISC_COLUMNS = [
    "class_label",
    "x0",
    "y0",
    "x1",
    "y1",
    "im_width",
    "im_height",
    "image_name",
    "low_quality", # stats required for the filtering step
    "cytoplasm_area",
    "nucleus_area",
    "nucleus_compactness",
    "nucleus_roundness",
    "entire_area",
    "entire_compactness",
    "entire_roundness",
]

ALL_RESULTS = [
    "vacuolated",
    "mds_asynchrony",
    "mds_dysmorphic",
    "mds_multinucleated",
    "mitotic",
    "apl_auer",
    "apl_bilobed",
    "apl_hypergranulated",
    "mgk_normal",
    "mgk_multinucleated",
    "mgk_hypolobated",
    "mgk_broken",
    "mgk_unclear",
    "mgk_sufficient",
]

APL_RESULTS = [
    "apl_hypergranulated",
    "apl_bilobed",
    "apl_auer",
]

APL_CUTOFFS = {
    "apl_auer": 0.75,
    "apl_bilobed": 0.65,
    "apl_hypergranulated": 0.50,
}

APL_INCLUDED_CLASSES = {
    "Blasts": 2,
    "Promyelocytes": 16,
    "Myelocytes": 11,
    "APL": CELL_LINEAGES["APL"],
}

MDS_RESULTS = [
    "mds_asynchrony",
    "mds_dysmorphic",
    "mds_multinucleated",
]

MDS_CUTOFFS = {
    "mds_asynchrony": 0.60,
    "mds_dysmorphic": 0.50,
    "mds_multinucleated": 0.50,
}

MDS_INCLUDED_CLASSES = {
    "Erythroblasts": 5,
}

MGK_NUCLEUS_RESULTS = [
    "mgk_multinucleated",
    "mgk_hypolobated",
]

MGK_NUCLEUS_CUTOFFS = {
    "mgk_multinucleated": 0.50,
    "mgk_hypolobated": 0.60,
}

MGK_QUALITY_FILTER_CLASSES = [
    "mgk_broken",
    "mgk_unclear",
    "mgk_sufficient",
]

MGK_INCLUDED_CLASSES = {
    "Megakaryocytes": 8,
}

MGK_AGGREGATE_RULES = {
    "All": {
        "Numerator": [
            "mgk_hypolobated",
            "mgk_multinucleated",
        ],
        "Denominator": "Megakaryocytes",
    },
    "Sufficient": {
        "Numerator": [
            "mgk_hypolobated",
            "mgk_multinucleated",
        ],
        "Denominator": "Megakaryocytes-mgk_sufficient",
    }
}

MGK_NON_MODELING_FEATURES = [
    "Megakaryocytes-mgk_broken_proportion",
    "Megakaryocytes-mgk_unclear_proportion",
    "Megakaryocytes-mgk_sufficient_proportion",
    "Megakaryocytes-All-mgk_hypolobated_proportion",
    "Megakaryocytes-All-mgk_multinucleated_proportion",
]

# Overall dysplasia
MYELOID_DYSPLASIA = {
    2: [
        ("apl_auer", 0.75),
        ("apl_bilobed", 0.65),
        ("apl_hypergranulated", 0.50),
        ("vacuolated", 0.50)
    ],
    16: [
        ("apl_auer", 0.75),
        ("apl_bilobed", 0.65),
        ("apl_hypergranulated", 0.50),
        ("vacuolated", 0.50)
    ],
    11: [
        ("apl_hypergranulated", 0.50),
        ("vacuolated", 0.50)
    ],
}

ERYTHROID_DYSPLASIA = {
    5: [
        ("mds_asynchrony", 0.60),
        ("mds_dysmorphic", 0.50),
        ("mds_multinucleated", 0.50),
        ("vacuolated", 0.50)
    ]   
}
CELL_LABEL_MAP = {
    0: "Artefacts",
    1: "Band_neutrophils",
    2: "Basophils",
    3: "Blasts",
    4: "Eosinophils",
    5: "Erythroblasts",
    6: "Giant_platelets",
    7: "Lymphocytes",
    8: "Metamyelocytes",
    9: "Monocytes",
    10: "Myelocytes",
    11: "Plasma_cells",
    12: "Platelet_aggregations",
    13: "Segmented_neutrophils",
    14: "Background" # RBCs + other detector hallucinations
}

REV_CELL_LABEL_MAP = {v: k for k, v in CELL_LABEL_MAP.items()}

CELL_STATISTICS_CLASSES = [
    1, # Band_neutrophils
    2, # Basophils
    3, # Blasts
    4, # Eosinophils
    5, # Erythroblasts
    7, # Lymphocytes
    8, # Metamyelocytes
    9, # Monocytes
    10, # Myelocytes
    11, # Plasma_cells
    13, # Segmented_neutrophils
]

ERYTHROBLAST_QUALITY_CLASSES = [
    5, # Erythroblasts
]

GRANULOCYTE_QUALITY_CLASSES = [
    1, # Band_neutrophils
    2, # Basophils
    4, # Eosinophils
    8, # Metamyelocytes
    10, # Myelocytes
    13, # Segmented_neutrophils
]

CELL_LINEAGES = {
    "Living_cells": [
        1, # Band_neutrophils
        2, # Basophils
        3, # Blasts
        4, # Eosinophils,
        5, # Erythroblasts,
        7, # Lymphocytes,
        8, # Metamyelocytes,
        9, # Monocytes
        10, # Myelocytes,
        11, # Plasma_cells
        13, # Segmented_neutrophils
    ],
    "All_detections": [
        1, # Band_neutrophils
        2, # Basophils
        3, # Blasts
        4, # Eosinophils,
        5, # Erythroblasts,
        6, # Giant_platelets
        7, # Lymphocytes,
        8, # Metamyelocytes,
        9, # Monocytes
        10, # Myelocytes,
        11, # Plasma_cells
        12, # Platelet_aggregations
        13, # Segmented_neutrophils
    ],
    "Granulopoetic_cells": [
        1, # Band_neutrophils
        2, # Basophils
        4, # Eosinophils
        8, # Metamyelocytes
        10, # Myelocytes
        13, # Segmented_neutrophils
    ],
    "Monocytic_cells": [
        9, # Monocytes
    ],
    "Erythropoetic_cells": [
        5, # Erythroblasts
    ],
    "Lymphoid_cells": [
        7, # Lymphocytes
        11, # Plasma_cells
    ],
    "Granulocytes": [
        1, # Band_neutrophils
        2, # Basophils
        4, # Eosinophils
        13, # Segmented_neutrophils
    ],
    "APL": [
        3, # Blasts
        10, # Myelocytes
    ],
    "Abnormal_platelets": [
        6, # Giant_platelets
        12, # Platelet_aggregations
    ],
    "Neutrophils": [
        1, # Band_neutrophils
        13, # Segmented_neutrophils
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
    "image_name",
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
    "low_quality",
    "image_name",
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
    "macropolycyte",
    "neut_gran_normal",
    "neut_gran_hypo",
    "neut_gran_hyper",
    "neut_segm_normal",
    "neut_segm_hypo",
    "neut_segm_hyper",
]

APL_RESULTS = [
    "apl_auer",
    "apl_bilobed",
    "apl_hypergranulated"
]

APL_CUTOFFS = {
    "apl_auer": 0.75,
    "apl_bilobed": 0.65,
    "apl_hypergranulated": 0.50,
}

APL_INCLUDED_CLASSES = {
    "Blasts": 3,
    "Myelocytes": 12,
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

MACROP_RESULTS = [
    "macropolycyte"
]

MACROP_CUTOFFS = {
    "macropolycyte": 0.5,
}

MACROP_INCLUDED_CLASSES = {
    "Band_neutrophils": 1, 
    "Segmented_neutrophils": 13, 
    "Neutrophils": CELL_LINEAGES["Neutrophils"]
}

# Separate from macrop, as multiclass (3-way)
NEUT_RESULTS = [
    [
    "neut_gran_normal",
    "neut_gran_hypo",
    "neut_gran_hyper",
    ],
    [
    "neut_segm_normal",
    "neut_segm_hypo",
    "neut_segm_hyper",
    ]
]

NEUT_INCLUDED_CLASSES = {
    "Band_neutrophils": 1, 
    "Segmented_neutrophils": 13, 
    "Neutrophils": CELL_LINEAGES["Neutrophils"]
}

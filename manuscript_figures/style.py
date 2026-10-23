"""Shared figure constants: target order and colours, cell-type names and lineage colours, effect-size scale."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 5, "pdf.fonttype": 42})
MM, W = 1 / 25.4, 177  # inch per mm; full page width (mm)

GROUPS = [("APL", "#a92b27", ["PML-RARA"]),
          ("CBF fusions", "#b95a1e", ["CBFB-MYH11", "RUNX1-RUNX1T1_OR_t_8_21"]),
          ("Epigenetic", "#9b9231", ["IDH_mutations", "IDH2", "CpG_mutations"]),
          ("NPM1", "#455fa2", ["NPM1", "NPM1_AND_FLT3_ITD"]),
          ("Signaling / RTK-RAS", "#4e867a", ["FLT3_mutations", "FLT3_ITD", "RTK_RAS_mutations", "KIT"]),
          ("Secondary AML", "#5f4480", ["complex_karyotype", "TP53", "monosomal_karyotype", "monosomy5_del_5q", "STAG2"])]
VARS = [v for _, _, vs in GROUPS for v in vs]
VNAME = {"RUNX1-RUNX1T1_OR_t_8_21": "RUNX1-RUNX1T1", "IDH_mutations": "IDH mutations", "CpG_mutations": "CpG mutations",
         "NPM1_AND_FLT3_ITD": "NPM1 + FLT3-ITD", "FLT3_mutations": "FLT3 mutations", "FLT3_ITD": "FLT3-ITD",
         "RTK_RAS_mutations": "RTK/RAS mutations", "complex_karyotype": "Complex karyotype",
         "monosomal_karyotype": "Monosomal karyotype", "monosomy5_del_5q": "Monosomy 5 / del(5q)"}

LINEAGE_COL = {"Blasts": "#dc6777", "Erythropoiesis": "#dbd182", "Megakaryocytes": "#8F8360",
               "Immature granulopoiesis": "#85cd85", "Granulocytes": "#dcb082", "Monocytopoiesis": "#a483d6",
               "Lymphopoiesis": "#88b5de", "Myeloid precursors": "#f8d097"}
CT_LINEAGE = {"Blasts": "Blasts", "Proerythroblasts": "Erythropoiesis", "Erythroblasts": "Erythropoiesis",
              "Megakaryocytes": "Megakaryocytes", "Promyelocytes": "Immature granulopoiesis",
              "Myelocytes": "Immature granulopoiesis", "Metamyelocytes": "Immature granulopoiesis",
              "Eosinophils_immature": "Immature granulopoiesis", "Eosinophils": "Granulocytes", "Basophils": "Granulocytes",
              "Neutrophils": "Granulocytes", "Band_neutrophils": "Granulocytes", "Segmented_neutrophils": "Granulocytes",
              "Promonocytes": "Monocytopoiesis", "Monocytes": "Monocytopoiesis", "Lymphocytes": "Lymphopoiesis",
              "Plasma_cells": "Lymphopoiesis", "Myeloid_precursors": "Myeloid precursors"}
CT_NAME = {"Eosinophils_immature": "Immature eosinophils", "Eosinophils": "Mature eosinophils",
           "Plasma_cells": "Plasma cells", "Myeloid_precursors": "Myeloid precursors",
           "Band_neutrophils": "Band neutrophils", "Segmented_neutrophils": "Seg. neutrophils"}
CT_SHORT = {"Proerythroblasts": "Proerythrobl.", "Eosinophils_immature": "Imm. eos.", "Eosinophils": "Mat. eos.",
            "Metamyelocytes": "Metamyel.", "Promyelocytes": "Promyel.", "Promonocytes": "Promono.",
            "Band_neutrophils": "Band neutr.", "Segmented_neutrophils": "Seg. neutr."}
DIFF_ROWS = {"BM": ["Blasts", "Proerythroblasts", "Erythroblasts", "Promyelocytes", "Myelocytes", "Metamyelocytes",
                    "Eosinophils_immature", "Eosinophils", "Basophils", "Neutrophils", "Promonocytes", "Monocytes",
                    "Lymphocytes", "Plasma_cells"],
             "PB": ["Blasts", "Erythroblasts", "Myelocytes", "Metamyelocytes", "Eosinophils", "Basophils",
                    "Band_neutrophils", "Segmented_neutrophils", "Monocytes", "Lymphocytes", "Plasma_cells"]}
MORPH_CT = {"BM": ["Myeloid_precursors", "Blasts", "Proerythroblasts", "Erythroblasts", "Megakaryocytes", "Promyelocytes",
                   "Myelocytes", "Metamyelocytes", "Eosinophils_immature", "Eosinophils", "Basophils", "Neutrophils",
                   "Promonocytes", "Monocytes", "Lymphocytes", "Plasma_cells"],
            "PB": ["Myeloid_precursors", "Blasts", "Erythroblasts", "Myelocytes", "Metamyelocytes", "Eosinophils",
                   "Basophils", "Neutrophils", "Band_neutrophils", "Segmented_neutrophils", "Monocytes", "Lymphocytes"]}
FEAT_NAME = {"hypergranulated": "hypergranular", "bilobed": "bilobed nucleus", "auer": "Auer rods",
             "vacuolated": "vacuolated", "asynchrony": "N:C asynchrony", "dysmorphic": "dysmorphic",
             "multinucleated": "separated nuclei", "hypolobated": "hypolobated nuclei", "apl_hypergranulated": "hypergranular",
             "neut_gran_normal": "normal granularity", "neut_gran_hypo": "hypogranular", "neut_gran_hyper": "hypergranular"}

OUTS = ["TN", "FN", "FP", "TP"]
OUT_COL = {"TN": "#3a6fb0", "FN": "#f39c34", "FP": "#d62728", "TP": "#8b1a1a"}

cmap = plt.get_cmap("coolwarm"); norm = matplotlib.colors.Normalize(-1, 1)  # Cliff's δ


def vname(v):
    return VNAME.get(v, v)


def vcol(v):
    return next(c for _, c, vs in GROUPS if v in vs)


def ctname(c):
    return CT_NAME.get(c, c.replace("_", " "))


def lineage_col(ct):
    return LINEAGE_COL.get(CT_LINEAGE.get(ct, ""), "#bbbbbb")


def axes_mm(fig, x, y, w, h, H, width=W):
    """Axes placed in mm from the top-left corner of a width × H mm figure."""
    return fig.add_axes([x / width, 1 - (y + h) / H, w / width, h / H])


def size_of(p):
    """Balloon diameter (pt) for an FDR-adjusted p: 2.6 at 0.05 → 6.8 at ≤ 0.001."""
    return 2.6 + 4.2 * (np.clip(-np.log10(np.clip(p, 1e-3, 1)), 1.3, 3) - 1.3) / 1.7

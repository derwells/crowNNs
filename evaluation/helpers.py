import os

EVAL_CSV = "data/evaluation/RGB/benchmark_annotations.csv"
EVAL_ROOT = "data/evaluation/RGB"
SITES = [
    'ABBY', 'ARIK', 'BARC', 'BARR', 'BART', 'BIGC', 'BLAN',
    'BLDE', 'BLUE', 'BLWA', 'BONA', 'CARI', 'CLBJ', 'COMO',
    'CPER', 'CRAM', 'CUPE', 'DCFS', 'DEJU', 'DELA', 'DSNY',
    'FLNT', 'GRSM', 'GUAN', 'GUIL', 'HARV', 'HEAL', 'HOPB',
    'JERC', 'JORN', 'KING', 'KONA', 'KONZ', 'LAJA', 'LECO',
    'LENO', 'LEWI', 'LIRO', 'MART', 'MAYF', 'MCDI', 'MCRA',
    'MLBS', 'MOAB', 'NIWO', 'NOGP', 'OAES', 'OKSR', 'ONAQ',
    'ORNL', 'OSBS', 'POSE', 'PRIN', 'PRLA', 'PRPO', 'PUUM',
    'REDB', 'RMNP', 'SCBI', 'SERC', 'SJER', 'SOAP', 'SRER',
    'STEI', 'STER', 'SUGG', 'SYCA', 'TALL', 'TEAK', 'TECR',
    'TOMB', 'TOOK', 'TOOL', 'TREE', 'UKFS', 'UNDE', 'WALK',
    'WLOU', 'WOOD', 'WREF', 'YELL'
]


def get_models_to_test(model_dir):
    models_to_test = os.listdir(model_dir)
    models_to_test = [
        os.path.join(model_dir, m_file) for m_file in models_to_test
    ]

    return models_to_test


def add_site_field(df):
    patterns = '|'.join(SITES)
    extracted_sites = df['image_path'].str.extract(f"({patterns})", expand=False)
    df['site'] = extracted_sites

    return df


def f1_score(p, r):
    return 2 * p * r / (p + r)

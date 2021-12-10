import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))

MINIO_KEYS = os.path.join(PROJECT_PATH, 'minio.json')
LOGS_PATH = os.path.join(PROJECT_PATH, 'logs')
DETECTION_LOGS = os.path.join(LOGS_PATH, 'detection_coords_inference')
PRED_MAP_TABLE_LOGS = os.path.join(LOGS_PATH, 'pred_map_to_table')

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
TIF_PATH = os.path.join(DATA_PATH, 'tif_data')
COORDS_PATH = os.path.join(DATA_PATH, 'label_coordinates')
CROPS_PATH = os.path.join(DATA_PATH, 'atom_crops_data')
PROBS_PATH = os.path.join(DATA_PATH, 'probs_data')
BOX_PATH = os.path.join(DATA_PATH, 'box_data')
PREDS_PATH = os.path.join(DATA_PATH, 'prediction_cache')
DETECTION_PATH = os.path.join(DATA_PATH, 'detection_data')

DATASET_PATH = os.path.join(PROJECT_PATH, 'dataset')
CROPS_DATASET = os.path.join(DATASET_PATH, 'atom_crops.csv')
PROBS_DATASET = os.path.join(DATASET_PATH, 'probs_dataset.csv')
BF_DATASET = os.path.join(DATASET_PATH, 'BF_dataset.csv')
HAADF_DATASET = os.path.join(DATASET_PATH, 'HAADF_dataset.csv')
COORDS_DATASET = os.path.join(DATASET_PATH, 'Coordinate_image_pairs.csv')
BOX_DATASET = os.path.join(DATASET_PATH, 'box_dataset.csv')

MODELS_PATH = os.path.join(PROJECT_PATH, 'models')

DATA_VIS_PATH = os.path.join(PROJECT_PATH, 'data_vis')
CROPS_VIS_PATH = os.path.join(DATA_VIS_PATH, 'crops')
CM_VIS_PATH = os.path.join(DATA_VIS_PATH, 'cm_vis')
ORIG_VIS_PATH = os.path.join(DATA_VIS_PATH, 'orig')
LABEL_VIS_PATH = os.path.join(DATA_VIS_PATH, 'label')
PRED_VIS_PATH = os.path.join(DATA_VIS_PATH, 'predictions')
PRED_GT_VIS_PATH = os.path.join(DATA_VIS_PATH, 'predictions_gt')
LANDS_VIS_PATH = os.path.join(DATA_VIS_PATH, 'landscapes')

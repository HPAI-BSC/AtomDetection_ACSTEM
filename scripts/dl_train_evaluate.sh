CURRENT_PATH=$(dirname "$0")
PROJECT_PATH=$(dirname $CURRENT_PATH)
PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
echo $PROJECT_PATH
cd $PROJECT_PATH
PYTHONPATH=$PROJECT_PATH python atoms_detection/dl_full_pipeline.py replicate4 basic dataset/Coordinate_image_pairs.csv -t 0.89
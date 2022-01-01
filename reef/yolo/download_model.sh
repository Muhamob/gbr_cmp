model_name=${1:-yolov5s.pt}
echo "Attempting download ${model_name} model"
python -c "from utils.downloads import attempt_download; attempt_download('$model_name')"

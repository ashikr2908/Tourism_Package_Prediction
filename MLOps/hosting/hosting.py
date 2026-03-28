from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="MLOps/deployment",     # Corrected path to deployment files
    repo_id="ashikr/tourist_package_prediction",          # Target Hugging Face Space
    repo_type="space",                      # repo type is space for hosting apps
    path_in_repo="",                          
)

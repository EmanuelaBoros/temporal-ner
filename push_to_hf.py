from huggingface_hub import Repository, HfFolder
import os
from shutil import copytree, rmtree


def upload_model_to_huggingface(model_directory, repo_id):
    """
    Uploads a model directory to an existing Hugging Face Hub repository.

    Args:
    model_directory (str): Local path to the model directory.
    repo_id (str): User or organization name and repository name on Hugging Face.
    """
    # Authentication token
    token = HfFolder.get_token()
    if token is None:
        raise ValueError(
            "Hugging Face token not found. Please log in using `huggingface-cli login`."
        )

    # Define the local directory for cloning
    local_clone_dir = "hf_repo_clone"
    if os.path.exists(local_clone_dir):
        rmtree(local_clone_dir)

    # Clone the repository from Hugging Face Hub
    repo = Repository(
        local_dir=local_clone_dir, clone_from=repo_id, use_auth_token=token
    )
    print("Repository cloned successfully.")

    # Ensure the cloned directory is ready
    if not os.path.exists(local_clone_dir):
        os.makedirs(local_clone_dir)

    # Determine the target directory within the cloned repo (directly in root)
    target_model_dir = os.path.join(local_clone_dir, os.path.basename(model_directory))

    # Copy the model to the cloned repository root
    copytree(model_directory, target_model_dir)
    print(f"Model {os.path.basename(model_directory)} copied to repository.")

    # Stage all files in git and push them to the repository
    repo.git_add(auto_lfs_track=True)
    commit_message = f"Add/update model in repository"
    repo.git_commit(commit_message)
    repo.git_push()

    print(
        f"Model '{os.path.basename(model_directory)}' has been pushed to the repository: {repo_id}"
    )


# Usage example
model_directory = "models_store/ner"
repo_id = "impresso-project/bert-impresso-ner-multilingual"  # Replace with your Hugging Face username and repository name

upload_model_to_huggingface(model_directory, repo_id)

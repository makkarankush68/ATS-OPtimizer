import os
import logging
from typing import List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
from scripts.utils.logger import init_logging_config

init_logging_config(basic_log_level=logging.INFO)
# Get the logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)


def find_path(folder_name):
    """
    The function `find_path` searches for a folder by name starting from the current directory and
    traversing up the directory tree until the folder is found or the root directory is reached.

    Args:
      folder_name: The `find_path` function you provided is designed to search for a folder by name
    starting from the current working directory and moving up the directory tree until it finds the
    folder or reaches the root directory.

    Returns:
      The `find_path` function is designed to search for a folder with the given `folder_name` starting
    from the current working directory (`os.getcwd()`). It iterates through the directory structure,
    checking if the folder exists in the current directory or any of its parent directories. If the
    folder is found, it returns the full path to that folder using `os.path.join(curr_dir, folder_name)`
    """
    curr_dir = os.getcwd()
    while True:
        if folder_name in os.listdir(curr_dir):
            return os.path.join(curr_dir, folder_name)
        else:
            parent_dir = os.path.dirname(curr_dir)
            if parent_dir == "/":
                break
            curr_dir = parent_dir
    raise ValueError(f"Folder '{folder_name}' not found.")


cwd = find_path("Resume-Matcher")
READ_RESUME_FROM = os.path.join(cwd, "Data", "Processed", "Resumes")
READ_JOB_DESCRIPTION_FROM = os.path.join(cwd, "Data", "Processed", "JobDescription")
config_path = os.path.join(cwd, "scripts", "similarity")

def get_score(resume_string, job_description_string):
    model = SentenceTransformer(
        "paraphrase-MiniLM-L6-v2"
    )  # Efficient model for similarity
    resume_embedding = model.encode(resume_string)
    job_description_embedding = model.encode(job_description_string)

    # Using pytorch_cos_sim for cosine similarity
    similarity = util.pytorch_cos_sim(resume_embedding, job_description_embedding)[0][0]
    print(similarity)
    # other method to get general score
    documents: List[str] = [resume_string]
    client = QdrantClient(":memory:")
    client.set_model("BAAI/bge-base-en")

    client.add(
        collection_name="demo_collection",
        documents=documents,
    )

    search_result = client.query(
        collection_name="demo_collection", query_text=job_description_string
    )
    return [similarity, search_result]


# if __name__ == "__main__":
#     # To give your custom resume use this code
#     resume_dict = read_config(
#         READ_RESUME_FROM
#         + "/Resume-alfred_pennyworth_pm.pdf83632b66-5cce-4322-a3c6-895ff7e3dd96.json"
#     )
#     job_dict = read_config(
#         READ_JOB_DESCRIPTION_FROM
#         + "/JobDescription-job_desc_product_manager.pdf6763dc68-12ff-4b32-b652-ccee195de071.json"
#     )
#     resume_keywords = resume_dict["extracted_keywords"]
#     job_description_keywords = job_dict["extracted_keywords"]

#     resume_string = " ".join(resume_keywords)
#     jd_string = " ".join(job_description_keywords)
#     final_result = get_score(resume_string, jd_string)
#     for r in final_result:
#         print(r.score)

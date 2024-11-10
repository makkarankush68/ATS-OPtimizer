import os
import json
import nltk
import logging
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from typing import List
from streamlit_extras import add_vertical_space as avs
from scripts import JobDescriptionProcessor, ResumeProcessor
from scripts.utils import init_logging_config
from annotated_text import annotated_text
from scripts.utils import get_filenames_from_dir
from scripts.similarity.get_score import get_score

# Initialize logging configuration
init_logging_config()

# Define paths
PROCESSED_RESUMES_PATH = "Data\Processed\Resumes"
PROCESSED_JOB_DESCRIPTIONS_PATH = "Data\Processed\JobDescription"
RESUMES_PATH = "Data\Resumes"
JOB_DESCRIPTIONS_PATH = "Data\JobDescription"


# Ensure directories exist
def ensure_directories_exist(paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")


# Create necessary directories
ensure_directories_exist(
    [
        PROCESSED_RESUMES_PATH,
        PROCESSED_JOB_DESCRIPTIONS_PATH,
        RESUMES_PATH,
        JOB_DESCRIPTIONS_PATH,
    ]
)


# Utility functions
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Remove old files
def remove_old_files(files_path):
    if not os.path.exists(files_path):
        os.makedirs(files_path)
    for filename in os.listdir(files_path):
        file_path = os.path.join(files_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    logging.info(f"Deleted old files from {files_path}")


# Tokenize and annotate text
def create_annotated_text(
    input_string: str, word_list: List[str], annotation: str, color_code: str
):
    # Tokenize the input string
    tokens = nltk.word_tokenize(input_string)

    # Convert the list to a set for quick lookups
    word_set = set(word_list)

    # Initialize an empty list to hold the annotated text
    annotated_text = []

    for token in tokens:
        # Check if the token is in the set
        if token in word_set:
            # If it is, append a tuple with the token, annotation, and color code
            annotated_text.append((" " + token, annotation, color_code))
        else:
            # If it's not, just append the token as a string
            annotated_text.append(" " + token)

    return annotated_text


# Create star graph
def create_star_graph(nodes_and_weights, title):
    # Create an empty graph
    G = nx.Graph()

    # Add the central node
    central_node = "resume"
    G.add_node(central_node)

    # Add nodes and edges with weights to the graph
    for node, weight in nodes_and_weights:
        G.add_node(node)
        G.add_edge(central_node, node, weight=weight * 100)

    # Get position layout for nodes
    pos = nx.spring_layout(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Rainbow",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    # Color node points by number of connections
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.adj[node])  # changes here
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"{node}<br># of connections: {len(adjacencies)}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Show the figure
    st.plotly_chart(fig)


# Start Streamlit app
st.title("ResCraft - ATS Optimizer")
st.header("Upload your resume and job description to get insights on ATS optimization")

# Resume upload section
st.subheader("Step 1: Upload Your Resume (PDF)")
resume_file = st.file_uploader("Upload your resume in PDF format", type="pdf")
if resume_file:
    remove_old_files(PROCESSED_RESUMES_PATH)
    remove_old_files(RESUMES_PATH)
    # Remove old files before processing a new one
    resume_path = os.path.join(RESUMES_PATH, resume_file.name)
    with open(resume_path, "wb") as f:
        f.write(resume_file.getbuffer())
    st.success(f"Resume uploaded successfully: {resume_path}")

    resume_processor = ResumeProcessor(resume_path)
    resume_processor.process()  # Process and save the resume in JSON format

# Job description text input
st.subheader("Step 2: Paste the Job Description")
job_description = st.text_area("Paste the job description here")
if job_description:
    remove_old_files(PROCESSED_JOB_DESCRIPTIONS_PATH)
    remove_old_files(JOB_DESCRIPTIONS_PATH)  # Clear old files
    job_description_path = os.path.join(JOB_DESCRIPTIONS_PATH, "job_description.pdf")
    print(job_description_path)

    pdf = FPDF(format="A4")  # Specify A4 page size
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    job_description = job_description.replace("\u2019", "'")
    pdf.multi_cell(0, 10, txt=job_description)

    pdf.output(job_description_path)
    job_description_processor = JobDescriptionProcessor(job_description_path)
    job_description_processor.process()  # Process and save the job description


# Display job descriptions and resumes
resume_names = get_filenames_from_dir(PROCESSED_RESUMES_PATH)
st.markdown(f"##### {len(resume_names)} resumes available. Please select one:")
selected_resume = st.selectbox("", resume_names, index=0)
if selected_resume:
    selected_file = read_json(os.path.join(PROCESSED_RESUMES_PATH, selected_resume))
    st.markdown("#### Parsed Resume Data")
    st.caption("Parsed text from your resume:")
    # st.write(selected_file["clean_data"])
    annotated_text(
        create_annotated_text(
            selected_file["clean_data"],
            selected_file["extracted_keywords"],
            "KW",
            "#0B666A",
        )
    )
    st.write("Entities from the Resume:")
    create_star_graph(selected_file["keyterms"], "Entities from Resume")

# Job descriptions available
job_descriptions = get_filenames_from_dir(PROCESSED_JOB_DESCRIPTIONS_PATH)
st.markdown(
    f"##### {len(job_descriptions)} job descriptions available. Please select one:"
)
selected_jd_name = st.selectbox("", job_descriptions, index=1)
if selected_jd_name:
    selected_jd = read_json(
        os.path.join(PROCESSED_JOB_DESCRIPTIONS_PATH, selected_jd_name)
    )
    st.markdown("#### Job Description")
    # st.write(selected_jd["clean_data"])
    annotated_text(
        create_annotated_text(
            selected_jd["clean_data"],
            selected_jd["extracted_keywords"],
            "KW",
            "#0B666A",
        )
    )
    st.write("Entities from the job description:")
    create_star_graph(selected_jd["keyterms"], "Entities from Job Description")
    # Annotate common words and show similarity score
    annotated_text(
        create_annotated_text(
            selected_file["clean_data"],
            selected_jd["extracted_keywords"],
            "JD",
            "#F24C3D",
        )
    )
#! TODO:
# st.write("Entities from the job description:")
# create_star_graph(selected_jd["keyterms"], "Entities from Job Description")


# Similarity score
if selected_jd_name and selected_resume:
    resume_string = " ".join(selected_file["extracted_keywords"])
    jd_string = " ".join(selected_jd["extracted_keywords"])
    result = get_score(resume_string, jd_string)
    similarity_score = round(float(result * 100), 2)
    score_color = (
        "green"
        if similarity_score >= 75
        else "orange" if similarity_score >= 60 else "red"
    )
    st.markdown(
        f"Similarity Score: <span style='color:{score_color};font-size:24px;font-weight:bold'>{similarity_score}</span>",
        unsafe_allow_html=True,
    )

import os
import time
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
            annotated_text.append((" " + token + " ", annotation, color_code))
        else:
            # If it's not, just append the token as a string
            annotated_text.append(" " + token + " ")

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
        # Multiply edge weight by 30
        G.add_edge(central_node, node, weight=weight * 30)

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
            colorscale="Viridis",  # Use a color scale that reflects weights
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Edge Weights",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    # Normalize the edge weights to the range 0-30
    min_weight = (
        min(weight for node, weight in nodes_and_weights) * 30
    )  # after multiplying by 30
    max_weight = (
        max(weight for node, weight in nodes_and_weights) * 30
    )  # after multiplying by 30
    max_normalized = 30

    def normalize_weight(weight):
        # Normalize between 0 and 30
        return min(
            ((weight - min_weight) / (max_weight - min_weight)) * max_normalized,
            max_normalized,
        )

    # Color node points by the normalized weight of the edge connecting it to the central node
    node_weights = []
    node_text = []
    for node in G.nodes():
        if node != central_node:
            weight = G[central_node][node][
                "weight"
            ]  # Get the weight of the edge to the central node
            normalized_weight = normalize_weight(weight)
            node_weights.append(normalized_weight)
            node_text.append(
                f"{node}<br>Normalized Edge weight: {normalized_weight:.2f}"
            )
        else:
            node_weights.append(0)  # The central node has no edge weight
            node_text.append(f"{node}<br>Edge weight: 0")

    node_trace.marker.color = node_weights
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


# Page configuration
st.set_page_config(page_title="ResCraft - ATS Optimizer", page_icon="📄", layout="wide")


# Main title
st.markdown(
    """
    <h1 style='text-align: center; color: #E2E8F0; margin-bottom: 3rem; animation: fadeIn 1.5s;'>
        ResCraft - ATS Optimizer ✨
    </h1>
""",
    unsafe_allow_html=True,
)

# Create two columns for the main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
        <div style='background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%); 
                padding: 2rem; border-radius: 1rem; 
                box-shadow: 0 4px 15px -1px rgba(0,0,0,0.2);
                border: 1px solid rgba(96, 165, 250, 0.2);'>
            <h3 style='color: #60A5FA;'>📤 Upload Resume</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    if resume_file:
        try:
            with st.spinner("Processing resume..."):
                time.sleep(1)  # Simulate processing
                remove_old_files(PROCESSED_RESUMES_PATH)
                remove_old_files(RESUMES_PATH)
                resume_path = os.path.join(RESUMES_PATH, resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getbuffer())

                resume_processor = ResumeProcessor(resume_path)
                resume_processor.process()

                st.success("✅ Resume processed successfully!")
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")

with col2:
    st.markdown(
        """
        <div style='background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%); 
             padding: 2rem; border-radius: 1rem; 
             box-shadow: 0 4px 15px -1px rgba(0,0,0,0.2);
             border: 1px solid rgba(96, 165, 250, 0.2);'>
            <h3 style='color: #60A5FA;'>📝 Job Description</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    job_description = st.text_area("Paste job description here", height=150)
    if job_description:
        try:
            with st.spinner("Analyzing job description..."):
                time.sleep(1)  # Simulate processing
                remove_old_files(PROCESSED_JOB_DESCRIPTIONS_PATH)
                remove_old_files(JOB_DESCRIPTIONS_PATH)

                job_description_path = os.path.join(
                    JOB_DESCRIPTIONS_PATH, "job_description.pdf"
                )
                pdf = FPDF(format="A4")
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                job_description = job_description.replace("\u2019", "'")
                pdf.multi_cell(0, 10, txt=job_description)
                pdf.output(job_description_path)

                job_description_processor = JobDescriptionProcessor(
                    job_description_path
                )
                job_description_processor.process()

                st.success("✅ Job description analyzed!")
        except Exception as e:
            st.error(f"Error processing job description: {str(e)}")

# Results section
st.markdown(
    """
    <div style='background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%); 
            padding: 2rem; border-radius: 1rem; 
            box-shadow: 0 4px 15px -1px rgba(0,0,0,0.2);
            border: 1px solid rgba(96, 165, 250, 0.2);
            margin-top: 2rem;'>
        <h2 style='color: #E2E8F0; text-align: center; margin-bottom: 2rem;'>Analysis Results 📊</h2>
    </div>
""",
    unsafe_allow_html=True,
)

# Display parsed data
selected_resume = None
resume_names = get_filenames_from_dir(PROCESSED_RESUMES_PATH)
if resume_names:
    try:
        st.markdown("#### 🔑 Key Terms from Resume")
        selected_resume = resume_names[0]
        selected_file = read_json(os.path.join(PROCESSED_RESUMES_PATH, selected_resume))

        col5, col6 = st.columns([1, 1])

        with col5:
            with st.expander("📄 View Parsed Resume Data", expanded=True):
                annotated_text(
                    create_annotated_text(
                        selected_file["clean_data"],
                        selected_file["extracted_keywords"],
                        "",
                        "#60A5FA",
                    )
                )

        with col6:
            create_star_graph(selected_file["keyterms"], "Entities from Resume")
            df2 = pd.DataFrame(selected_file["keyterms"], columns=["keyword", "value"])
            fig = px.treemap(
                df2,
                path=["keyword"],
                values="value",
                color_continuous_scale="Rainbow",
                title="Key Terms/Topics Extracted from the selected Job Description",
            )
            st.write(fig)
    except Exception as e:
        st.error(f"Error displaying resume data: {str(e)}")

selected_jd_name = None
# Job description analysis
job_descriptions = get_filenames_from_dir(PROCESSED_JOB_DESCRIPTIONS_PATH)
if job_descriptions:
    try:
        st.markdown("#### 🎯 Key Terms from Job Description")
        selected_jd_name = job_descriptions[0]
        selected_jd = read_json(
            os.path.join(PROCESSED_JOB_DESCRIPTIONS_PATH, selected_jd_name)
        )

        col3, col4 = st.columns([1, 1])

        with col3:
            with st.expander("📋 View Parsed Job Description", expanded=True):
                annotated_text(
                    create_annotated_text(
                        selected_jd["clean_data"],
                        selected_jd["extracted_keywords"],
                        "",
                        "#60A5FA",
                    )
                )

        with col4:
            create_star_graph(selected_jd["keyterms"], "Entities from Job Description")
            df2 = pd.DataFrame(selected_jd["keyterms"], columns=["keyword", "value"])
            fig = px.treemap(
                df2,
                path=["keyword"],
                values="value",
                color_continuous_scale="Rainbow",
                title="Key Terms/Topics Extracted from the selected Job Description",
            )
            st.write(fig)
    except Exception as e:
        st.error(f"Error displaying job description data: {str(e)}")

# Similarity score section
if selected_jd_name and selected_resume:
    st.markdown("#### 🎯 Matched Keywords")

    with st.expander("📋 View Matched Keywords", expanded=True):
        annotated_text(
            create_annotated_text(
                selected_file["clean_data"],
                selected_jd["extracted_keywords"],
                "",
                "#F24C3D",
            )
        )
    try:
        resume_string = " ".join(selected_file["extracted_keywords"])
        jd_string = " ".join(selected_jd["extracted_keywords"])
        [result1, result2] = get_score(resume_string, jd_string)

        similarity_score1 = round(float(result1 * 100), 2)
        similarity_score2 = round(float(result2[0].score * 100), 2)

        # Updated header with more contrast and bolder design
        st.markdown(
            """
            <div style='background: linear-gradient(145deg, #0F172A 0%, #020617 100%); 
                    padding: 2.5rem; 
                    border-radius: 1rem; 
                    box-shadow: 0 8px 25px -2px rgba(0,0,0,0.3);
                    border: 2px solid rgba(96, 165, 250, 0.3);
                    margin-top: 2rem;'>
                <h2 style='color: #FFFFFF; 
                        text-align: center; 
                        font-weight: 700;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                    Match Analysis 🎯
                </h2>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Create two columns for the metrics
        metric_col1, metric_col2 = st.columns(2)

        # Helper function for status color
        def get_status_color(score):
            if score >= 75:
                return "#10B981"  # Green
            elif score >= 60:
                return "#F59E0B"  # Yellow
            return "#EF4444"  # Red

        # Create two columns for the metrics
        metric_col1, metric_col2 = st.columns(2)

        # Custom CSS for centering
        st.markdown(
            """
            <style>
                [data-testid="metric-container"] {
                    width: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    text-align: center;
                }
                .css-1wivap2 {
                    text-align: center;
                    width: 100%;
                }
                [data-testid="stMetricLabel"] {
                    width: 100%;
                    text-align: center;
                    justify-content: center;
                }
                [data-testid="stMetricValue"] {
                    width: 100%;
                    text-align: center;
                    justify-content: center;
                }
                [data-testid="stMetricDelta"] {
                    width: 100%;
                    text-align: center;
                    justify-content: center;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        with metric_col1:
            st.metric(
                label="Resume vs Job Description Match",
                value=f"{similarity_score1}%",
                delta=f"{'Excellent' if similarity_score1 >= 75 else 'Good' if similarity_score1 >= 60 else 'Needs Improvement'}",
                delta_color="normal" if similarity_score1 >= 60 else "inverse",
            )

        with metric_col2:
            st.metric(
                label="General Resume Score",
                value=f"{similarity_score2}%",
                delta=f"{'Excellent' if similarity_score2 >= 75 else 'Good' if similarity_score2 >= 60 else 'Needs Improvement'}",
                delta_color="normal" if similarity_score2 >= 60 else "inverse",
            )
        # Enhanced gauge charts with higher contrast
        fig = go.Figure()

        # Common gauge settings
        gauge_settings = {
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "#FFFFFF",
                "tickfont": {"size": 14, "color": "#FFFFFF"},
            },
            "bar": {"color": "skyblue", "thickness": 0.6},
            "bgcolor": "#1E293B",
            "borderwidth": 3,
            "steps": [
                {"range": [0, 60], "color": "red"},
                {"range": [60, 75], "color": "orange"},
                {"range": [75, 100], "color": "green"},
            ],
        }

        # Gauge for similarity_score1
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=similarity_score1,
                title={
                    "text": "Resume vs JD Score",
                    "font": {"size": 26, "color": "#FFFFFF", "family": "Arial Black"},
                },
                gauge=gauge_settings,
                domain={"x": [0, 0.5], "y": [0, 1]},
                number={
                    "font": {"color": "#FFFFFF", "size": 40, "family": "Arial Black"},
                    "suffix": "%",
                },
            )
        )

        # Gauge for similarity_score2
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=similarity_score2,
                title={
                    "text": "General Resume Score",
                    "font": {"size": 26, "color": "#FFFFFF", "family": "Arial Black"},
                },
                gauge=gauge_settings,
                domain={"x": [0.5, 1], "y": [0, 1]},
                number={
                    "font": {"color": "#FFFFFF", "size": 40, "family": "Arial Black"},
                    "suffix": "%",
                },
            )
        )

        # Update layout with higher contrast
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font={"color": "#FFFFFF"},
            height=500,
            margin=dict(t=100, b=100),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating similarity score: {str(e)}")

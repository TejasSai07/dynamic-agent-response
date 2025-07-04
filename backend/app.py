from flask import Flask, request, jsonify
from openai import AzureOpenAI
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS
import uuid
import re
from datetime import datetime
import sqlite3
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import numpy as np
import requests
import urllib.parse
import time
import json
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import os
import faiss
import networkx as nx
import pandas as pd
import io
import sys
import re
import json
import traceback
from typing import Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import base64
from sqlalchemy import Column, LargeBinary
from sqlalchemy import and_, or_


app = Flask(__name__, static_folder="static", static_url_path="/static")

CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)

app.config['CORS_HEADERS'] = 'Content-Type'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
db = SQLAlchemy(app)

import matplotlib.pyplot as plt


# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response


client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", "your-api-key-here"),
    api_version="2024-02-15-preview",
    azure_endpoint="https://radiusofself.openai.azure.com",
    azure_deployment="gpt-4o"
)


class AgentDefinition(db.Model):
    __tablename__ = 'agent_definitions'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type = db.Column(db.String(40), nullable=False)  # GPT-4o, Claude Sonet, etc.
    name = db.Column(db.String(100), nullable=False)
    system_prompt = db.Column(db.Text, nullable=False)
    tools = db.Column(db.Text, nullable=True)  # JSON string of tool configs
    memory_enabled = db.Column(db.Boolean, default=False)
    tasks = db.Column(db.Text, nullable=True)  # Optional default tasks
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<AgentDefinition {self.name} ({self.model_type})>'

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), nullable=False)
    agent_type = db.Column(db.String(40), nullable=False)  # 'data_analysis', 'knowledge_extraction', or 'custom'
    model_type = db.Column(db.String(40), nullable=False)  # 'gpt-4o', 'claude-sonnet', etc.
    agent_definition_id = db.Column(db.String(36), db.ForeignKey('agent_definitions.id'), nullable=True)
    role = db.Column(db.String(20), nullable=False)  # 'system', 'user', or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Fields for data analysis agent
    code = db.Column(db.Text, nullable=True)
    output = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    plot_path = db.Column(db.String(255), nullable=True)
    plot_paths = db.Column(db.Text, nullable=True)  # JSON string of plot paths
    pickled_objects = db.Column(db.LargeBinary, nullable=True)
    execution_history = db.Column(db.Text, nullable=True)
    
    # Fields for knowledge extraction agent
    tool_used = db.Column(db.String(100), nullable=True)
    tool_payload = db.Column(db.Text, nullable=True)
    tool_output = db.Column(db.Text, nullable=True)
    step_number = db.Column(db.Integer, nullable=True)
    
    # Relationship to agent definition
    agent_definition = db.relationship('AgentDefinition', backref='conversations', foreign_keys=[agent_definition_id])
    
    def save_objects(self, objects_dict):
        """Pickle and save Python objects"""
        try:
            # Filter out non-picklable objects (e.g., modules, functions)
            # This is a critical step to prevent pickling errors
            filtered_objects = {}
            for key, obj in objects_dict.items():
                try:
                    # Attempt to pickle a dummy object to check picklability
                    # This is a more robust check than type checking
                    pickle.dumps(obj)
                    filtered_objects[key] = obj
                except TypeError as e:
                    print(f"⚠️ Warning: Object '{key}' of type {type(obj)} is not picklable. Skipping. Error: {e}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not check picklability of object '{key}': {e}")


            if filtered_objects:
                print(f"📦 Pickling filtered objects: {filtered_objects.keys()}")
                self.pickled_objects = pickle.dumps(filtered_objects)
            else:
                self.pickled_objects = None
                print("No picklable objects to save.")

        except Exception as e:
            print(f"Error pickling objects: {e}")

    def load_objects(self):
        """Load pickled Python objects"""
        if not self.pickled_objects:
            return {}
        try:
            return pickle.loads(self.pickled_objects)
        except Exception as e:
            print(f"Error unpickling objects: {e}")
            return {}


@app.route('/api/conversation/start', methods=['POST'])
def start_conversation():
    """Starts a new conversation with the selected agent"""
    data = request.json
    agent_id = data.get('agent_id')
    print(agent_id)
    if not agent_id:
        return jsonify({"error": "Agent ID is required"}), 400

    conversation_id = str(uuid.uuid4())

    # Define tool access messages based on the agent type
    agent_tool_info = {
        "knowledge_extraction": (
            "Welcome to the Knowledge Extraction Agent! You have access to the following tools:\n\n"
            "🔹 **FAISS KNN** - Retrieve similar knowledge based on embeddings.\n"
            "🔹 **Knowledge Graph** - Explore relationships between concepts.\n"
            "🔹 **Web Search using FAISS KNN** - Find relevant information from indexed sources.\n\n"
            "How can I assist you?"
        ),
        "data_analysis": (
            "Welcome to the Data Analysis Agent! You have access to the **Code Executor**, which allows "
            "you to run Python code for data processing and analysis.\n\n"
            "How can I assist you?"
        )
    }

    # Get the appropriate welcome message, defaulting to a generic one
    welcome_message_text = agent_tool_info.get(agent_id, f"Welcome to the {agent_id} agent! How can I assist you?")
    definition = AgentDefinition.query.get(agent_id)

    if definition:
        agent_type = "custom"
        agent_definition_id = agent_id
    else:
        agent_type = agent_id  # 'data_analysis' or 'knowledge_extraction'
        agent_definition_id = None

    # Save the welcome message with correct agent type and definition
    welcome_message = ChatHistory(
        conversation_id=conversation_id,
        agent_type=agent_type,
        agent_definition_id=agent_definition_id,
        model_type=definition.model_type if definition else "gpt-4o",
        role="system",
        content=welcome_message_text
    )
    
    db.session.add(welcome_message)
    db.session.commit()

    return json.dumps({
        "conversation_id": conversation_id,
        "agent_id": agent_id,
        "message": welcome_message_text  # Include the message in the response
    }, cls=CustomJSONEncoder)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def strip_markdown(text):
    """
    Convert Markdown-like syntax to HTML-like text.
    """
    text = re.sub(r'(^|\n)#{6}\s*(.+)', r'<h6>\2</h6>', text)
    text = re.sub(r'(^|\n)#{5}\s*(.+)', r'<h5>\2</h5>', text)
    text = re.sub(r'(^|\n)#{4}\s*(.+)', r'<h4>\2</h4>', text)
    text = re.sub(r'(^|\n)#{3}\s*(.+)', r'<h3>\2</h3>', text)
    text = re.sub(r'(^|\n)#{2}\s*(.+)', r'<h2>\2</h2>', text)
    text = re.sub(r'(^|\n)#\s*(.+)', r'<h1>\2</h1>', text)

    # Convert bold/italic emphasis to <strong> and <em>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)  # Italics

    # Convert inline code (backticks) to <code>
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'<code>\1</code>', text)

    # Remove unnecessary whitespace and return clean HTML-like text
    return text.strip()


def safe_request(url, headers=None, max_retries=3, timeout=10):
    """
    Safely fetch a URL with retries and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx, 5xx)
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:  # Don't wait after the last attempt
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch {url} after multiple retries.")
                return None

@app.route('/api/agents', methods=['GET'])
def get_available_agents():
    """Returns a list of available agents"""
    agents = [
        {
            "id": "data_analysis",
            "name": "Data Analysis Agent",
            "description": "An agent specialized in analyzing data through code execution."
        },
        {
            "id": "knowledge_extraction",
            "name": "Knowledge Extraction Agent",
            "description": "An agent for extracting knowledge from documents and the web."
        }
    ]
    return json.dumps(agents, cls = CustomJSONEncoder)

# -----------------------------------------------------------------------------
# Text Processing Functions
# -----------------------------------------------------------------------------

#Load NLTK 
try:
    punkt_path = '/Users/TejasSai/nltk_data/tokenizers/punkt/english.pickle'
    with open(punkt_path, 'rb') as f:
        tokenizer = pickle.load(f)
except LookupError:
    punkt_path = '/Users/TejasSai/nltk_data/tokenizers/punkt/english.pickle'
    with open(punkt_path, 'rb') as f:
        tokenizer = pickle.load(f)


def read_pdf(file_path):
    """
    Extract text from a PDF file.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None

def split_into_paragraphs(text):
    """
    Split text into paragraphs.
    """
    paragraphs = []
    current_para = []

    # Split by double newlines
    rough_splits = text.split('\n\n')

    for split in rough_splits:
        # Further clean and process each split
        lines = split.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                continue
            current_para.append(line)

        if current_para:
            paragraphs.append(' '.join(current_para))
            current_para = []

    # Add any remaining paragraph
    if current_para:
        paragraphs.append(' '.join(current_para))

    # Filter out very short splits that might be artifacts
    paragraphs = [p for p in paragraphs if len(p.split()) > 5]

    return paragraphs


def chunk_text_by_paragraphs(text, max_chunk_size=512):
    """
    Chunk text into segments, trying to keep paragraphs intact.
    """
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        # Get sentences in this paragraph
        sentences = tokenizer.tokenize(paragraph)
        paragraph_size = len(paragraph.split())

        # If a single paragraph is larger than max_chunk_size,
        # we need to split it (though we'd prefer not to)
        if paragraph_size > max_chunk_size:
            # Process this large paragraph sentence by sentence
            temp_chunk = []
            temp_size = 0

            for sentence in sentences:
                sentence_size = len(sentence.split())
                if temp_size + sentence_size > max_chunk_size:
                    if temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                    temp_chunk = [sentence]
                    temp_size = sentence_size
                else:
                    temp_chunk.append(sentence)
                    temp_size += sentence_size

            if temp_chunk:
                chunks.append(" ".join(temp_chunk))

        # If adding this paragraph would exceed max_chunk_size,
        # save current chunk and start new one
        elif current_size + paragraph_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [paragraph]
            current_size = paragraph_size

        # Add paragraph to current chunk
        else:
            current_chunk.append(paragraph)
            current_size += paragraph_size

    # Add the last chunk if any
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for text chunks using SentenceTransformer.
    """
    try:
        model = SentenceTransformer(model_name)
        return model.encode(chunks, show_progress_bar=False)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def process_all_pdfs(directory_path):
    """
    Process all PDF files in a directory, chunk the text, and generate embeddings.
    """
    all_chunks = []
    embeddings = []
    file_mapping = {}

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")

            try:
                text = read_pdf(file_path)
                if not text:
                    print(f"Warning: No text extracted from {filename}")
                    continue

                chunks = chunk_text_by_paragraphs(text)
                if not chunks:
                    print(f"Warning: No chunks created from {filename}")
                    continue

                vectors = get_embeddings(chunks)
                if vectors is None or len(vectors) != len(chunks):
                    print(f"Warning: Mismatch between chunks and vectors for {filename}")
                    continue

                start_index = len(all_chunks)
                all_chunks.extend(chunks)
                embeddings.extend(vectors)
                file_mapping[filename] = {
                    'start_index': start_index,
                    'end_index': len(all_chunks) - 1
                }

                print(f"Processed {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if not all_chunks:
        raise ValueError("No chunks were created from any PDF files")

    embeddings = np.array(embeddings)
    return all_chunks, embeddings, file_mapping

# -----------------------------------------------------------------------------
# Initialize FAISS Index
# -----------------------------------------------------------------------------

# Load data and create embeddings
DIRECTORY_PATH = '/Users/TejasSai/Desktop/ML_Projects/BioMedical_Graph_Knowledge_Graphs/Directory_of_Files'  # Replace with your directory path
try:
    all_chunks, embeddings, file_mapping = process_all_pdfs(DIRECTORY_PATH)

    print(f"Total number of chunks across all PDFs: {len(all_chunks)}")
    print(f"Shape of all embeddings: {embeddings.shape}")
    print(f"Number of processed files: {len(file_mapping)}")

    # Initialize FAISS index
    EMBEDDING_DIM = embeddings.shape[1]
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    print(f"Number of embeddings in index: {index.ntotal}")

except Exception as e:
    print(f"Error initializing FAISS index: {e}")
    all_chunks, embeddings, file_mapping, index = [], [], {}, None

# -----------------------------------------------------------------------------
# Web Search and Article Extraction
# -----------------------------------------------------------------------------

def duckduckgo_search(query, max_results=5):
    """
    Perform a web search using DuckDuckGo.
    """
    search_url = f"https://www.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = safe_request(search_url, headers)
    if not response:
        print("Failed to fetch search results.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    search_results = []

    for result in soup.select(".result__title a")[:max_results]:
        raw_link = result["href"]

        # Extract actual URL from DuckDuckGo redirect
        parsed_link = urllib.parse.parse_qs(urllib.parse.urlparse(raw_link).query).get("uddg")
        if parsed_link:
            actual_url = parsed_link[0]  # Extract first element from list
            title = result.get_text()
            search_results.append((title, actual_url))

    return search_results


def extract_article_text(url):
    """
    Extract meaningful text from a web article.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = safe_request(url, headers)

    if not response:
        print(f"Failed to fetch content from {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Try multiple content selectors
    content_selectors = [
        ("div", "article-content"),
        ("article", None),
        ("div", "content"),
        ("div", "main-content"),
        ("div", "entry-content"),
        ("div", "post-content"),
        ("div", "article-body"),
        ("section", "content-section"),
        ("div", "story-body"),
        ("div", "news-content"),
        ("div", "news-article"),
        ("div", "article"),
        ("section", "article-body"),
        ("div", "article-text"),
        ("div", "story"),
        ("div", "news-body"),
        ("body", None)  # Last resort (entire page)
    ]

    unwanted_keywords = [
        "Sign in", "Log into your account", "Forgot your password?", "Recover your password",
        "Subscribe", "Get help", "Navigation", "Menu", "Search", "Welcome!", "Watch & Bet",
        "Follow us on", "Trending", "Latest news", "Social media", "Facebook", "Twitter", "Instagram", "Youtube"
    ]

    for tag, class_name in content_selectors:
        content_block = soup.find(tag, class_=class_name) if class_name else soup.find(tag)
        if content_block:
            extracted_text = content_block.get_text("\n", strip=True)

            # Remove lines containing unwanted keywords
            extracted_lines = [line for line in extracted_text.split("\n") if
                               not any(keyword in line for keyword in unwanted_keywords)]
            cleaned_text = "\n".join(extracted_lines)

            # Check if extracted content is meaningful (not just a few words)
            if len(cleaned_text.split()) > 50:  # Only return text if it's longer than 50 words
                return cleaned_text

    print("Could not find meaningful article content.")
    return None


# -----------------------------------------------------------------------------
# OpenAI Completion
# -----------------------------------------------------------------------------

def my_completion(messages):
    """Interact with Azure OpenAI to generate a completion."""
    client = AzureOpenAI(
        api_key="e353f4ba413e41fbb54023a915ae98e6",
        api_version="2024-02-15-preview",
        azure_endpoint="https://radiusofself.openai.azure.com",
        azure_deployment="gpt-4o"
    )
    chat_completion = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"} 
    )
    return chat_completion.choices[0].message.content


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def current_date() -> str:
    """
    Return current date as a string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------------------------------------------------------
# Database Functions
# -----------------------------------------------------------------------------

def initialize_database():
    """
    Initialize SQLite database with necessary tables.
    """
    conn = sqlite3.connect('context_database.db')
    cursor = conn.cursor()

    # Table for storing query-answer pairs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    # Table for storing web search results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS web_search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            results TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    conn.commit()
    conn.close()

def web_search(query_string: str, num_results: int) -> str:
    """
    Perform a web search using DuckDuckGo and cache results in SQLite.
    Returns formatted search results as a string.
    """
    conn = sqlite3.connect('context_database.db')
    cursor = conn.cursor()
    
    try:
        # Check if we have cached results
        cursor.execute('SELECT results FROM web_search_results WHERE query = ? AND timestamp > datetime("now", "-1 day")', (query_string,))
        cached_result = cursor.fetchone()
        
        if cached_result:
            conn.close()
            return cached_result[0]
        
        search_results = duckduckgo_search(query_string, max_results=num_results)

        if not search_results:
            return json.dumps({"error": "No search results found."})

        extracted_articles = []
        
        for title, link in search_results:
            article_text = extract_article_text(link)
            
            if article_text:
                extracted_articles.append({
                    "title": title,
                    "link": link,
                    "text": article_text # Limit to first 1000 characters to avoid excessive length
                })
        
        if not extracted_articles:
            return jsonify({"error": "Failed to extract relevant article content from search results."})

        # Convert results to JSON string before storing
        results_json = json.dumps({
            "source": "live_search",
            "search_results": extracted_articles
        })
        
        # Store the results in the database
        cursor.execute('''
            INSERT INTO web_search_results (query, results) 
            VALUES (?, ?)
        ''', (query_string, results_json))
        conn.commit()
        
        return results_json
    
    except Exception as e:
        print(f"Error in web_search: {e}")
        return json.dumps({"error": str(e)})  # Return error as JSON string
    
    finally:
        conn.close()


def initialize_database():
    """
    Initialize SQLite database with necessary tables.
    """
    conn = sqlite3.connect('context_database.db')
    cursor = conn.cursor()

    # Table for storing web search results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS web_search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            results TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

    conn.commit()
    conn.close()

def gathered_context_visualization(query_string: str, contexts: list, distances: list) -> str:
    """
    Create a visualization of context relevance using Plotly.
    Returns HTML string of the visualization.
    """
    try:
        # Create relevance scores (inverse of distances)
        relevance_scores = [1 / (1 + d) for d in distances]
        
        # Create truncated context labels
        context_labels = [f"{ctx[:50]}..." for ctx in contexts]
        
        # Create Plotly figure
        fig = go.Figure(data=[
            go.Bar(
                x=relevance_scores,
                y=context_labels,
                orientation='h',
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title=f'Context Relevance for Query: "{query_string}"',
            xaxis_title='Relevance Score',
            yaxis_title='Contexts',
            height=100 + (len(contexts) * 50),  # Dynamic height based on number of contexts
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Save visualization to HTML file with timestamp
        filename = f'Context_Visualizations/context_visualization_{query_string[:25]}.html'
        fig.write_html(filename)
        
        return f"Visualization saved as {filename}"
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

def extract_key_terms(query_string: str) -> str:
    """
    Extract and analyze key terms from the query using NLTK.
    Returns formatted string of key terms and their properties.
    """
    try:
        # Initialize NLTK tools
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Tokenize and process text
        tokens = word_tokenize(query_string.lower())
        
        # Remove stopwords and lemmatize
        key_terms = []
        pos_tags = nltk.pos_tag(tokens)
        
        for token, pos in pos_tags:
            if token not in stop_words and token.isalnum():
                lemma = lemmatizer.lemmatize(token)
                key_terms.append({
                    'term': token,
                    'lemma': lemma,
                    'pos': pos
                })
        
        # Format results
        if not key_terms:
            return "No key terms found"
            
        formatted_terms = "\n".join([
            f"Term: {t['term']}\n  Lemma: {t['lemma']}\n  Part of Speech: {t['pos']}"
            for t in key_terms
        ])
        
        return formatted_terms
        
    except Exception as e:
        return f"Error extracting key terms: {str(e)}"
    
def run_faiss_knn(tool_payload):
    """Enhanced FAISS KNN search with automatic visualization."""
    query_string = tool_payload['query_string']
    k = tool_payload['k']

    try:
        query_embedding = get_embeddings(query_string)
        if query_embedding is None:
            return {"error": "Failed to generate embeddings for the query."}

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = index.search(query_embedding, k)

        # Gather contexts
        contexts = []
        total_context = ""
        for i in range(k):
            relationship_text = all_chunks[indices[0][i]]
            contexts.append(relationship_text)
            total_context += f"{i + 1}. {relationship_text}\n"
            # print(f"Relationship: {relationship_text}")
            # print(f"Distance: {distances[0][i]}")

        # Create visualization automatically
        viz_result = gathered_context_visualization(
            query_string,
            contexts,
            distances[0].tolist()
        )

        return {
            "cached_result": False,
            "answer": total_context,
            "context": contexts,
            "distances": distances[0].tolist(),
            "visualization": viz_result
        }

    except Exception as e:
        print(f"Error in run_faiss_knn: {e}")
        return {"error": str(e)}

def load_knowledge_graph():
    """Load and parse the knowledge graph data."""
    try:
        with open('/Users/TejasSai/Desktop/ML_Projects/BioMedical_Graph_Knowledge_Graphs/cumulative_entities1.json', 'r') as f:
            kg_data = json.load(f)
            return kg_data
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        return None
def create_graph(entities, relationships):
    """Creates a networkx graph from the entities and relationships."""
    graph = nx.DiGraph()
    for entity_name, entity_id in entities.items():
        print(f" Entity Name and Enitity ID : {entity_name}, {entity_id}")
        graph.add_node(entity_id, name=entity_name)

    for rel in relationships:
        print(f"Relationship : {rel}")
        graph.add_edge(rel['source'], rel['target'], type=rel['type'], description=rel['description'])
    return graph

def find_all_paths(graph, start_node, end_node):
    """Finds all paths between two nodes in the graph."""
    try:
        all_paths = list(nx.all_simple_paths(graph, source=start_node, target=end_node))
        return all_paths
    except nx.NetworkXNoPath:
        return None
    except nx.NetworkXError as e:
        return f"Error: {str(e)}"


def llm_select_entities(query, entities):
    """
    Uses the LLM to select entities from the knowledge graph based on a query.
    """
    entity_names = list(entities.keys())
    entity_list_string = "\n".join(entity_names)

    prompt = f"""
    Given the following query: {query}, and the following list of entities from a knowledge graph:
    {entity_list_string}

    Which entities from the list are in the query? Remember to return entities in order of position w.r.t the query.  Return their names, separated by a comma.
    For example:  Entity A, Entity B, Entity C
    """

    messages = [{"role": "user", "content": prompt}]
    llm_output = my_completion(messages)
    print(f"LLM OUTPUT : {llm_output}")
    # Parse the LLM output to get the entity names
    try:
        entity_names = [name.strip() for name in llm_output.split(",")]
        print(f"Entity names : {entity_names}")
        #Look up the entity IDs
        entity_ids = [entities[name] for name in entity_names if name in entities]
        if len(entity_ids) >= 2:  # Require at least two entities for a path
            return entity_ids
        else:
            return None #Didn't get enough valid entities
    except:
        return None #Error parsing

def get_all_paths(query, kg_data):
    """
    Finds all paths between two entities in the knowledge graph,
    using the LLM to select the entities based on a query.
    """
    entities = kg_data['entities']
    relationships = kg_data['relationships']

    if not query:
        return {"error": "A query is required to select entities using the LLM."}

    # Use LLM to select the start and end
    selected_entities = llm_select_entities(query, entities)
    print(f"Selected Entities are  : {selected_entities}")
    if not selected_entities:
        return {"error": "Could not determine start and end entities from LLM."}

    # Find all paths between FIRST and LAST elements
    start_node, end_node = selected_entities[0], selected_entities[1]

    graph = create_graph(entities, relationships)
    all_paths = find_all_paths(graph, start_node, end_node)

    if all_paths:
        # Convert node IDs to names for better readability
        paths_with_details = []
        for path in all_paths:
            detailed_path = []
            for i in range(len(path) - 1):
                source = graph.nodes[path[i]]['name']
                target = graph.nodes[path[i + 1]]['name']
                edge_data = graph.get_edge_data(path[i], path[i + 1])
                relationship = edge_data.get('type', 'relationship')
                description = edge_data.get('description', 'No description')
                detailed_path.append(f"{source} --[{relationship}]--> {target}: {description}")
            paths_with_details.append(detailed_path)
        return {"all_paths": paths_with_details}
    else:
        return {"message": f"No paths found between {start_node} and {end_node}"}

def execute_knowledge_graph(tool_payload):
    """Execute the knowledge graph query and return structured results."""
    try:
        # Load knowledge graph data
        KG_DATA = load_knowledge_graph()
        if not KG_DATA:
            return "Error: Could not load knowledge graph data."
        query = tool_payload['query_string']
        result = get_all_paths(query, KG_DATA)
        
        # Format results
        print(f"Query: {query}")
        if "all_paths" in result:
            print("All Paths:")
            for path in result["all_paths"]:
                print(" -> ".join(path))
        else:
            print(f"Result: {result}")

        formatted_results = {
        'query': query,
        'type': 'paths' if 'all_paths' in result else 'result',
        'data': {}
    }
    
        if 'all_paths' in result:
            # Format each path as a joined string
            formatted_paths = []
            for path in result['all_paths']:
                formatted_path = " -> ".join(path)
                formatted_paths.append(formatted_path)
            
            formatted_results['data'] = {
                'paths': formatted_paths,
                'path_count': len(formatted_paths)
            }
        else:
            formatted_results['data'] = result
        
        return json.dumps(formatted_results)  # Serialize to JSON string
        
    except Exception as e:
        return f"Error executing knowledge graph query: {str(e)}"


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route('/')
def home():
    return "Chat API is running"

def handle_other_tools(tool_id, tool_payload):
    """Helper function to handle other tools."""
    if tool_id == 'current_date':
        return current_date()
    elif tool_id == 'web_search':
        return web_search(tool_payload['query_string'], tool_payload['num_results'])
    elif tool_id == 'extract_key_terms':
        return extract_key_terms(tool_payload['query_string'])
    elif tool_id == 'knowledge_graph':
        return execute_knowledge_graph(tool_payload)
    return None

@app.route('/api/conversations/<conversation_id>/messages', methods=['POST'])
def send_message(conversation_id):
    """Send a message to an agent"""
    try:
        print(f"📨 Received request for conversation: {conversation_id}")
        data = request.json or {}
        print(f"🧾 Request data: {data}")

        message = data.get('message', '').strip()
        file_paths = data.get('file_paths', [])
        
        preserved_objects = {}
        last_memory_state = ChatHistory.query.filter(
            ChatHistory.conversation_id == conversation_id,
            ChatHistory.pickled_objects.isnot(None)
        ).order_by(ChatHistory.timestamp.desc()).first()

        if last_memory_state:
            print("🧠 Found and loaded a previous state from the database.")
            preserved_objects = last_memory_state.load_objects()
        else:
            print("🧠 No previous state found. Starting fresh.")

        valid_paths = [p for p in file_paths if p and os.path.exists(p)]
        if valid_paths:
            print(f"📂 Loading new files for this turn: {valid_paths}")
            for file_path in valid_paths:
                try:
                    content, content_type = load_file_by_type(file_path)
                    # Create clean variable names without file extensions
                    file_name = os.path.basename(file_path)
                    clean_name = os.path.splitext(file_name)[0]  # Remove extension
                    
                    # Use a more user-friendly variable name
                    if content_type == 'df':
                        var_name = f"df_{clean_name}" if not clean_name.startswith('df') else clean_name
                    else:
                        var_name = f"{content_type}_{clean_name}"
                    
                    preserved_objects[var_name] = content
                    print(f"📊 Loaded {content_type} as variable: {var_name}")
                except Exception as e:
                    print(f"⚠️ Error loading file {file_path}: {e}")
        
        print(f"🔄 Current state includes objects: {list(preserved_objects.keys())}")

        # Fetch agent info
        agent_query = db.session.query(
            ChatHistory.agent_type,
            ChatHistory.model_type,
            ChatHistory.agent_definition_id
        ).filter_by(conversation_id=conversation_id).first()

        if not agent_query:
            return jsonify({"error": "Agent not found"}), 404

        agent_type, model_type, agent_definition_id = agent_query

        # Save user message
        user_message = ChatHistory(
            conversation_id=conversation_id,
            agent_type=agent_type,
            model_type=model_type,
            agent_definition_id=agent_definition_id,
            role="user",
            content=message
        )
        db.session.add(user_message)
        db.session.commit()

        # Handle based on agent type
        if agent_type == 'data_analysis':
            result = handle_data_analysis(conversation_id, message, valid_paths[0] if valid_paths else None)

        elif agent_type == 'knowledge_extraction':
            result = handle_knowledge_extraction(conversation_id, message)

        elif agent_type == 'custom' and agent_definition_id:
            definition = AgentDefinition.query.get(agent_definition_id)
            if not definition:
                return jsonify({"error": "Agent definition not found"}), 404

            # Forward only the first file for now (extendable later)
            result = handle_custom_agent(conversation_id, message, definition,
                                         preserved_objects=preserved_objects)
        else:
            return jsonify({"error": "Unsupported agent type"}), 400

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@dataclass
class AgentState:
    data: dict
    def __init__(self, initial_data):
        self.data = initial_data


def handle_custom_agent(conversation_id, message, definition, file_path=None, preserved_objects={}):
    try:
        print(f"Starting custom agent handler for conversation {conversation_id}")

        plot_paths_accumulated = []
        system_prompt = definition.system_prompt
        
        system_prompt += (
            "\n\nIMPORTANT: Your response MUST be a JSON object. "
            "Your main goal is to solve the user's request step-by-step. "
            "If you need to write and execute code, provide it in the 'code' field. "
            "After execution, I will provide you with the output and errors. "
            "Based on that feedback, you will plan your next step."
            "🔒 SYSTEM REQUIREMENT 🔒 : FIRST UNDERSTAND WHAT IS IN THE OBJECT for e.g if DF then do df.columns(), df.head(), df.describe() etc."
            "🔒 SYSTEM REQUIREMENT 🔒 : You MUST re-use the returned_objects dictionary already present in memory if it exists, instead of reinitializing it. Do not use returned_objects = {} if one is already available."
            "🔒 SYSTEM REQUIREMENT 🔒 : When saving plots, you MUST save them to the 'static' folder using os.makedirs('static', exist_ok=True') AND record the file path into returned_objects['plot_paths']"
            "🔒 SYSTEM REQUIREMENT 🔒 : Make sure no spaces are there in the file path"
            "🔒 SYSTEM REQUIREMENT 🔒 : Make sure the plot paths are added to returned_objects using returned_objects.setdefault('plot_paths', []).append(path)"
            "🔒 SYSTEM REQUIREMENT 🔒 : VERY IMPORTANTTTTTT: If you want to print, you can't think it acts like a notebook. YOU HAVE TO PRINT IT by calling print and other relevant functions!"
            "🔒 SYSTEM REQUIREMENT 🔒 : This is NOT a notebook — previous code cells do NOT persist. All functions and variables must be redefined or explicitly carried forward using `'persist_state': true`"
            "\n\nJSON Structure:"
            "{\n"
            "  \"reasoning\": \"Your thought process and explanation for EACH step. ***DO NOT use 'updated_answer' or any other key. This is the only supported explanation key***\",\n"
            "  \"code\": \"The Python code to execute for this step. (can be an empty string)\",\n"
            "  \"persist_state\": boolean (Set to true ONLY when you want to save the current variables for future turns),\n"
            "  \"is_complete\": boolean (Set to true ONLY when the final answer is ready and all tasks are done.)\n"
            "}\n"
        ) 
    
        messages = [{"role": "system", "content": system_prompt}]

        if preserved_objects:
            available_vars = ", ".join(preserved_objects.keys())
            initial_state_message = (
                f"The following variables have been restored/loaded and are available for use: {available_vars}. "
                "These are already loaded in memory - DO NOT reload them from files. "
                "**IMPORTANT AND REQUIRED** - Start by inspecting them (e.g., with `.head()`, `.columns`, `.info()`) to understand their structure before proceeding."
            )
            messages.append({"role": "system", "content": initial_state_message})
            
        conversation_history = ChatHistory.query.filter(
            ChatHistory.conversation_id == conversation_id,
            ChatHistory.role.in_(["user", "assistant"])
        ).order_by(ChatHistory.timestamp).all()

        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})

        not_complete = True
        final_answer = None
        reasoning_steps = []
        count = 0
        max_iterations = 15

        state = AgentState(initial_data=preserved_objects)
        
        while not_complete and count < max_iterations:
            count += 1
            raw_output = get_model_completion(messages, definition.model_type)

            try:
                output_json = json.loads(raw_output)
            except json.JSONDecodeError:
                # Handle JSON parsing error
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": "Error: Your last response was not valid JSON. Please correct it and adhere to the specified JSON format."})
                continue
            print(f"🔍 Model output JSON: {output_json}")
            is_complete = output_json.get('is_complete', False)
            reasoning = output_json.get("reasoning") or output_json.get("updated_answer", "")
            code_to_execute = output_json.get('code', '')
            should_persist_state = output_json.get('persist_state', False)
            
            # Initialize execution result variables
            execution_output, execution_error = "", ""
            execution_result = None
            plot_paths = []
            
            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": raw_output})
            
            if code_to_execute:
                print("⚙️ Executing code...")
                print(f"Code to execute:\n{code_to_execute}")
                execution_result = execute_code(code_to_execute, state)
                print(f'Execution result is : {execution_result}')
                execution_output = execution_result.output
                execution_error = execution_result.error
                # Collect all plot paths over multiple iterations
                persisted_returned = preserved_objects.get("returned_objects", {})
                persisted_paths = persisted_returned.get("plot_paths", [])
                new_paths = execution_result.returned_objects.get("plot_paths", [])
                merged_paths = list(dict.fromkeys(persisted_paths + new_paths))
                persisted_returned["plot_paths"] = merged_paths
                preserved_objects["returned_objects"] = persisted_returned
                plot_paths_accumulated = merged_paths
                print(plot_paths_accumulated)
                # Update the main state with any new or modified variables from the execution
                preserved_objects.update(state.data) 
                print(f"State updated. Current variables: {list(preserved_objects.keys())}")
                
                # Provide feedback to the model
                feedback = (
                    f"Your code has been executed.\n"
                    f"Output (stdout):\n---\n{execution_output or 'No output'}\n---\n"
                    f"Error:\n---\n{execution_error or 'None'}\n---\n"
                )
                if execution_error:
                    feedback += "The code failed. Please analyze the error and provide the corrected code. Do NOT repeat the failed code."
                else:
                    feedback += "The code executed successfully. Please proceed to the next step or conclude the task."
                messages.append({"role": "user", "content": feedback})
                
                if should_persist_state and definition.memory_enabled:
                    print(f"💾 Persisting state for conversation {conversation_id} as requested by the agent.")
                    memory_message = ChatHistory(
                        conversation_id=conversation_id,
                        agent_type="custom",
                        model_type=definition.model_type,
                        agent_definition_id=definition.id,
                        role="system",
                        content=f"State persisted. Variables saved: {list(preserved_objects.keys())}"
                    )
                    memory_message.save_objects(preserved_objects)
                    db.session.add(memory_message)
                    db.session.commit()
                    print("✅ State saved successfully.")

            # Prepare response content with safe defaults
            response_content = {
                "updated_answer": reasoning,
                "code": code_to_execute,
                "is_complete": is_complete,
                "output": execution_output,
                "error": execution_error,
                "plot_paths": plot_paths_accumulated
            }
            ai_message = ChatHistory(
                conversation_id=conversation_id,
                agent_type="custom",
                model_type=definition.model_type,
                agent_definition_id=definition.id,
                role="assistant",
                content=json.dumps(response_content),
                code=code_to_execute,
                step_number=count
            )

            db.session.add(ai_message)
            db.session.commit()

            reasoning_steps.append({
                "step_number": count,
                "reasoning": reasoning,
                "next_step": output_json.get('next_step', ''),
                "code": code_to_execute,
                "output": execution_output,
                "error": execution_error,
                "plot_paths": plot_paths_accumulated
            })
            print(reasoning)
            if is_complete:
                if not plot_paths_accumulated:
                    final_returned = preserved_objects.get("returned_objects", {})
                    if "plot_paths" in final_returned:
                        plot_paths_accumulated.extend(final_returned["plot_paths"])
                final_answer = {
                    "reasoning": "FINAL ANSWER:\n\n" + reasoning,
                    "code_to_execute": code_to_execute,
                    "final_output": execution_output,
                    "plot_paths": plot_paths_accumulated
                }
                not_complete = False

        # Final state persistence (only if memory is enabled and we have objects to save)
        if definition.memory_enabled and preserved_objects and not should_persist_state:
            memory_message = ChatHistory(
                conversation_id=conversation_id,
                agent_type="custom",
                model_type=definition.model_type,
                agent_definition_id=definition.id,
                role="system",
                content="Memory state updated at end of conversation"
            )
            memory_message.save_objects(preserved_objects)
            db.session.add(memory_message)
            db.session.commit()

        print(f"Custom agent handler completed. Is complete: {not not_complete}")
        
        result = {
            "reasoning_steps": reasoning_steps,
            "final_output": final_answer,
            "preserved_objects": list(preserved_objects.keys()) if preserved_objects else [],
            "plot_paths": plot_paths_accumulated
        }
        print('Plot Paths:', plot_paths_accumulated)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        print(f"Error in handle_custom_agent: {e}")
        return jsonify({"error": f"Error in custom agent: {str(e)}"}), 500
    
def handle_data_analysis(conversation_id, message, file_path=None):
    """Handle messages for the data analysis agent"""
    try:
        # Check conversation state to see if we have a DataFrame already
        conversation_state = db.session.query(ChatHistory).filter_by(
            conversation_id=conversation_id,
            agent_type='data_analysis'
        ).all()
        
        is_new_conversation = len(conversation_state) <= 1
        
        # Only require file path for new conversations
        if is_new_conversation and not file_path:
            error_message = ChatHistory(
                conversation_id=conversation_id,
                agent_type='data_analysis',
                model_type = 'gpt-4o',
                role="system",
                content="Please provide a CSV file to analyze."
            )
            db.session.add(error_message)
            db.session.commit()
            return json.dumps({
                "response": "Please provide a CSV file to analyze."
            })
            
        # Save user message to database
        user_message = ChatHistory(
            conversation_id=conversation_id,
            agent_type='data_analysis',
            model_type = 'gpt-4o',
            role="user",
            content=message
        )
        db.session.add(user_message)
        db.session.commit()
        
        # Handle file loading if file_path is provided (initial or subsequent file loads)
        if file_path:
            try:
                # Check if multiple CSV files are provided (comma-separated paths)
                if "," in file_path:
                    file_paths = [path.strip() for path in file_path.split(",")]
                    dataframes = []
                    
                    for path in file_paths:
                        try:
                            df_new = pd.read_csv(path)
                            # Add source file information as a column
                            filename = os.path.basename(path)
                            df_new['source_file'] = filename
                            dataframes.append(df_new)
                        except Exception as e:
                            return jsonify({
                                "error": f"Error reading CSV file {path}: {str(e)}"
                            }), 500
                    
                    # Initialize df as the first DataFrame to ensure it exists
                    df = dataframes[0]
                    
                    # Run the agent with the message and all dataframes (will be handled in the agent's first step)
                    result = run_agent(message, dataframes,conversation_id)
                else:
                    # Single CSV file
                    df = pd.read_csv(file_path)
                    result = run_agent(message, df,conversation_id)
            except FileNotFoundError:
                error_message = ChatHistory(
                    conversation_id=conversation_id,
                    agent_type='data_analysis',
                    model_type = 'gpt-4o',
                    role="system",
                    content=f"File not found at {file_path}"
                )
                db.session.add(error_message)
                db.session.commit()
                
                return jsonify({
                    "error": f"File not found at {file_path}"
                }), 400
            except Exception as e:
                error_message = ChatHistory(
                    conversation_id=conversation_id,
                    agent_type='data_analysis',
                    model_type = 'gpt-4o',
                    role="system",
                    content=f"Error reading CSV file: {str(e)}"
                )
                db.session.add(error_message)
                db.session.commit()
                
                return jsonify({
                    "error": f"Error reading CSV file: {str(e)}"
                }), 500
        else:
            # No file path provided but we have existing dataframes from previous interactions
            # We'll continue the conversation by running the agent with the message only
            result = run_agent(message, None,conversation_id)  # df will be retrieved from the state

        # Process the execution history to extract reasoning steps for frontend
        reasoning_steps = []
        plot_paths = []
        final_reasoning = ""
        final_code = ""
        
        if isinstance(result, dict) and 'execution_history' in result:
            for i, execution in enumerate(result['execution_history']):
                # Create a standardized step that the frontend expects
                reasoning = execution.get('reasoning', '')
                next_step = execution.get('next_step', '')
                code = execution.get('code_to_execute', '')
                
                step = {
                    "step_number": i + 1,
                    "reasoning": reasoning,
                    "next_step": next_step,
                    "code": code,
                }
                
                # Keep track of the last code executed
                if code:
                    final_code = code
                objects_to_pickle = {}
                # Add execution results if available
                if 'result' in execution:
                    step["output"] = execution['result'].get('output', '')
                    step["error"] = execution['result'].get('error', '')
                    
                    # Extract any plot paths
                    if 'returned_objects' in execution['result']:
                        returned_objects = execution['result']['returned_objects']
                        for key, value in returned_objects.items():
                            # Skip certain types that shouldn't be pickled
                            if not isinstance(value, (plt.Figure, str)) and key != 'plot_path':
                                # Add to objects that should be pickled
                                try:
                                    # Test if object is picklable
                                    pickle.dumps(value)
                                    objects_to_pickle[key] = value
                                except (pickle.PickleError, TypeError):
                                    # Skip objects that can't be pickled
                                    pass
                        if 'plot_path' in returned_objects:
                            plot_path = returned_objects['plot_path']
                            step["plot_path"] = plot_path
                            plot_paths.append(plot_path)
                
                reasoning_steps.append(step)
                
                # Store each step as a separate message in the database
                step_message = ChatHistory(
                    conversation_id=conversation_id,
                    agent_type='data_analysis',
                    model_type='gpt-4o',
                    role="assistant",
                    content=reasoning,
                    code=code,
                    output=step.get("output", ""),
                    error=step.get("error", ""),
                    plot_path=step.get("plot_path", ""),
                    plot_paths=json.dumps(plot_paths),
                    execution_history=json.dumps({"execution": execution}, cls=CustomJSONEncoder)
                )
                if objects_to_pickle:
                    step_message.save_objects(objects_to_pickle)
                db.session.add(step_message)
                
                # Track final reasoning (from steps marked as complete)
                if execution.get('is_complete') and reasoning:
                    final_reasoning = reasoning
        
        # If no step was explicitly marked complete, use the last reasoning
        if not final_reasoning and reasoning_steps:
            final_reasoning = reasoning_steps[-1]["reasoning"]
        
        # Commit all changes to database
        db.session.commit()
        
        # Format the response to match what the frontend expects for data_analysis
        response_data = {
            "response": final_reasoning or str(result),
            "reasoning_steps": reasoning_steps,
            "plot_paths": plot_paths,
            "final_output": {
                "reasoning": "",
                "status": result.get("status", ""),
                "iterations": result.get("iterations", 0)
            }
        }

        return json.dumps(response_data, cls=CustomJSONEncoder)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@dataclass
class ExecutionResult:
    output: str
    error: Optional[str] = None
    traceback: Optional[str] = None
    returned_objects: Dict[str, Any] = None


class OutputCapture:
    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class AgentState:
    def __init__(self, initial_data: Dict[str, Any]):
        self.data = initial_data
        self.execution_history = []

    def update(self, new_data: Dict[str, Any]):
        self.data.update(new_data)

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def add_execution(self, execution_data):
        self.execution_history.append(execution_data)


def extract_code(response: str) -> Optional[str]:
    match = re.search(r'``````', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_json_from_string(llm_output: str):
    # First try to find JSON in markdown code blocks
    pattern = r'``````'
    match = re.search(pattern, llm_output)

    
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass  # Continue if JSON decoding fails
    
    # If no markdown blocks found, try to find raw JSON
    try:
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, llm_output)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    
    print("Could not extract valid JSON from response")
    return None



def print_separator(title: str = None):
    """Print a separator line with optional title for better log readability."""
    width = 80
    if title:
        print(f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}")
    else:
        print(f"\n{'=' * width}")


def pretty_print_json(data: Dict[str, Any]):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2, default=str))


def inspect_dataframe(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(3).to_dict('records'),
        "null_counts": df.isnull().sum().to_dict()
    }


@app.route('/api/agent-definitions', methods=['GET'])
def get_agent_definitions():
    """Get all agent definitions"""
    try:
        # Get built-in agents (data_analysis and knowledge_extraction)
        built_in_agents = [
            {
                "id": "data_analysis",
                "name": "Data Analysis Agent",
                "description": "An agent specialized in analyzing data through code execution.",
                "model_type": "gpt-4o",  # Default model for built-in agents
                "is_built_in": True
            },
            {
                "id": "knowledge_extraction",
                "name": "Knowledge Extraction Agent",
                "description": "An agent for extracting knowledge from documents and the web.",
                "model_type": "gpt-4o",  # Default model for built-in agents
                "is_built_in": True
            }
        ]
        
        # Get custom agents from database
        custom_agents = []
        definitions = AgentDefinition.query.all()
        for definition in definitions:
            custom_agents.append({
                "id": definition.id,
                "name": definition.name,
                "description": f"Custom agent using {definition.model_type}",
                "model_type": definition.model_type,
                "memory_enabled": definition.memory_enabled,
                "is_built_in": False
            })
        
        # Combine both types
        all_agents = built_in_agents + custom_agents
        return json.dumps(all_agents, cls=CustomJSONEncoder)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent-definitions', methods=['POST'])
def create_agent_definition():
    """Create a new agent definition"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['model_type', 'name', 'system_prompt']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create new agent definition
        new_definition = AgentDefinition(
            model_type=data['model_type'],
            name=data['name'],
            system_prompt=data['system_prompt'],
            tools=json.dumps(data.get('tools', [])),
            memory_enabled=data.get('memory_enabled', False),
            tasks=json.dumps(data.get('tasks', []))
        )
        
        db.session.add(new_definition)
        db.session.commit()
        
        return jsonify({
            "id": new_definition.id,
            "message": "Agent definition created successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations/agent/<agent_id>', methods=['GET'])
def get_agent_conversations(agent_id):
    """Get all conversations for a specific agent"""
    try:
        # For built-in agents
        if agent_id in ['data_analysis', 'knowledge_extraction']:
            # Query conversations by agent_type
            conversation_ids = db.session.query(ChatHistory.conversation_id).filter_by(
                agent_type=agent_id
            ).distinct().all()
        else:
            # For custom agents, query by agent_definition_id
            conversation_ids = db.session.query(ChatHistory.conversation_id).filter_by(
                agent_definition_id=agent_id
            ).distinct().all()
        conversations = []
        for conv_id in conversation_ids:
            # Get the first user message as preview
            first_msg = ChatHistory.query.filter_by(
                conversation_id=conv_id[0],
                role='user'
            ).order_by(ChatHistory.timestamp.asc()).first()
            
            # Get the timestamp of conversation start
            start_time = ChatHistory.query.filter_by(
                conversation_id=conv_id[0]
            ).order_by(ChatHistory.timestamp.asc()).first()
            
            if first_msg and start_time:
                conversations.append({
                    "id": conv_id[0],
                    "preview": first_msg.content[:50] + "..." if len(first_msg.content) > 50 else first_msg.content,
                    "timestamp": start_time.timestamp.isoformat() if hasattr(start_time, 'timestamp') else None
                })
        # Sort conversations by timestamp (newest first)
        conversations.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True)
        
        return jsonify(conversations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/agent-definitions/<definition_id>', methods=['PUT'])
def update_agent_definition(definition_id):
    """Update an agent definition"""
    try:
        # Can't update built-in agents
        if definition_id in ['data_analysis', 'knowledge_extraction']:
            return jsonify({"error": "Cannot update built-in agents"}), 400
            
        definition = AgentDefinition.query.get(definition_id)
        if not definition:
            return jsonify({"error": "Agent definition not found"}), 404
            
        data = request.json
        
        # Update fields if provided
        if 'model_type' in data:
            definition.model_type = data['model_type']
        if 'name' in data:
            definition.name = data['name']
        if 'system_prompt' in data:
            definition.system_prompt = data['system_prompt']
        if 'tools' in data:
            definition.tools = json.dumps(data['tools'])
        if 'memory_enabled' in data:
            definition.memory_enabled = data['memory_enabled']
        if 'tasks' in data:
            definition.tasks = json.dumps(data['tasks'])
            
        db.session.commit()
        
        return jsonify({"message": "Agent definition updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agent-definitions/<definition_id>', methods=['DELETE'])
def delete_agent_definition(definition_id):
    """Delete an agent definition"""
    print(definition_id)
    try:
        # Can't delete built-in agents
        if definition_id in ['data_analysis', 'knowledge_extraction']:
            return jsonify({"error": "Cannot delete built-in agents"}), 400
            
        definition = AgentDefinition.query.get(definition_id)
        if not definition:
            return jsonify({"error": "Agent definition not found"}), 404
            
        db.session.delete(definition)
        db.session.commit()

        response = jsonify({"message": "Agent definition deleted successfully"})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def execute_code(code: str, state: AgentState) -> ExecutionResult:
    import os
    import uuid
    import json
    import matplotlib.pyplot as plt
    import textwrap

    plot_save_code = textwrap.dedent("""
    import os
    import matplotlib.pyplot as plt
    import uuid

    static_dir = os.path.join(os.getcwd(), "backend", "static")
    os.makedirs(static_dir, exist_ok=True)
    plot_path = os.path.join(static_dir, f"plot_{uuid.uuid4()}.png")
    web_path = f"/static/{os.path.basename(plot_path)}"
    plt.savefig(plot_path)
    returned_objects.setdefault("plot_paths", []).append(web_path)
    plt.close()
    """)

    code = code.replace("plt.show()", plot_save_code)

    if not code.strip():
        return ExecutionResult(output="No code to execute", returned_objects={})

    shared_env = state.data.copy()
    returned_objects = {}
    shared_env["returned_objects"] = returned_objects

    with OutputCapture() as output:
        try:
            if "\n" not in code and not code.strip().endswith(":"):
                result = eval(code, shared_env, shared_env)
                if result is not None:
                    returned_objects["result"] = str(result)
            else:
                exec(code, shared_env, shared_env)
                for k, v in shared_env.items():
                    if k not in state.data and k not in ["returned_objects"]:
                        returned_objects[k] = v

            # Catch plots that didn’t use plt.show()
            if plt.get_fignums():
                fallback_path = f"static/plot_{uuid.uuid4()}.png"
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                plt.savefig(fallback_path)
                plt.close()
                returned_objects.setdefault("plot_paths", []).append("/" + fallback_path)

            # Update shared state with serializable values
            for key, value in shared_env.items():
                if key not in state.data or state.data[key] is not value:
                    try:
                        json.dumps(value, default=str)
                        state.data[key] = value
                    except (TypeError, ValueError):
                        continue

            return ExecutionResult(
                output=output.stdout.getvalue(),
                error=None,
                traceback=None,
                returned_objects=returned_objects
            )

        except Exception as e:
            return ExecutionResult(
                output=output.stdout.getvalue(),
                error=str(e),
                traceback=traceback.format_exc(),
                returned_objects=returned_objects
            )


def run_agent(input_question: str, data=None, conversation_id = None):
    # Initialize agent state based on what's provided
    initial_state = AgentState({
        'pd': pd,
        'plt': plt,
        'inspect_dataframe': inspect_dataframe
    })
    previous_messages = []
    if conversation_id:
        # Try to find the most recent message with pickled objects
        latest_message = ChatHistory.query.filter_by(
            conversation_id=conversation_id,
            agent_type='data_analysis'
        ).filter(ChatHistory.pickled_objects != None).order_by(
            ChatHistory.timestamp.desc()
        ).first()
        
        if latest_message:
            # Load the objects and add them to the state
            objects = latest_message.load_objects()
            if objects:
                for key, value in objects.items():
                    initial_state.data[key] = value
                print(f"✅ Restored objects for conversation {conversation_id}: {list(objects.keys())}")
                
        history = ChatHistory.query.filter_by(
            conversation_id=conversation_id,
            agent_type='data_analysis'
        ).order_by(ChatHistory.timestamp.asc()).all()
        
        # Convert to message format expected by LLM
        for msg in history:
            if msg.role in ["user", "assistant"]:
                content = msg.content
                # For assistant messages that contained code, include the code and output
                if msg.role == "assistant" and msg.code:
                    content += f"\nCode executed:\n```python\n{msg.code}\n```"
                    if msg.output:
                        content += f"\nOutput:\n{msg.output}"
                
                previous_messages.append({
                    "role": msg.role,
                    "content": content
                })
    
    # Handle multiple dataframes or existing dataframe case
    if data is not None:
        if isinstance(data, list):
            # Multiple dataframes case - store all dataframes and set df to the first one initially
            initial_state.data['dataframes'] = data
            initial_state.data['df'] = data[0]
            # Flag to indicate multiple dataframes are available
            initial_state.data['multiple_dfs'] = True
        else:
            # Single dataframe case
            initial_state.data['df'] = data
            initial_state.data['multiple_dfs'] = False
    else:
        # No dataframe provided - this is a follow-up question
        # We'll add instructions for the LLM to continue with existing dataframes
        initial_state.data['continue_analysis'] = True

    # Add specific system prompt based on the state
    if initial_state.data.get('multiple_dfs', False):
        system_prompt = f'''
        You are an expert data scientist analyzing data through code execution.
        You have been provided with MULTIPLE CSV files which are available as a list called 'dataframes'.
        The first dataframe is already set as 'df' for convenience. Each dataframe has a 'source_file' column 
        indicating which file it came from.
        
        Available objects and functions:
        - dataframes: A list containing all loaded DataFrames
        - df: The first DataFrame (for convenience)
        - inspect_dataframe(df): Returns key DataFrame information
        - pd: pandas library
        - plt: matplotlib.pyplot
        
        IMPORTANT GUIDELINES:
        1. ALWAYS start by inspecting all dataframes using inspect_dataframe()
        2. Depending on the user's question, you might need to:
           - Merge/concatenate/join multiple dataframes
           - Analyze them separately and compare results
           - Focus on just one of the dataframes
        3. Break down complex tasks into steps
        4. Handle errors and missing data appropriately
        5. Return any created objects that need to be preserved
        
        Your output at each iteration must be a JSON with these fields:
        {{
            "reasoning": "Explain your thought process and what you learned",
            "next_step": "Clearly state what you will do next",
            "code_to_execute": "Your code here",
            "is_complete": false,
            "objects_to_preserve": ["list", "of", "variable", "names"]
        }}
        Do NOT return plain text responses, only JSON.
        '''
    elif initial_state.data.get('continue_analysis', False):
        system_prompt = f'''
        You are an expert data scientist analyzing data through code execution.
        This is a FOLLOW-UP question in an ongoing analysis. All previous objects and dataframes from 
        the prior conversation are still available in your execution context.
        
        Available objects and functions:
        - All variables from the previous conversation
        - Dataframes, plots, and results from previous analysis
        - inspect_dataframe(df): Returns key DataFrame information
        - pd: pandas library
        - plt: matplotlib.pyplot
        
        IMPORTANT GUIDELINES:
        1. First, check the existing objects to understand what's available
        2. You can reference and use all variables created in previous steps
        3. Break down complex tasks into steps
        4. Handle errors and missing data appropriately
        5. Return any created objects that need to be preserved
        
        Your output at each iteration must be a JSON with these fields:
        {{
            "reasoning": "Explain your thought process and what you learned",
            "next_step": "Clearly state what you will do next",
            "code_to_execute": "Your code here",
            "is_complete": false,
            "objects_to_preserve": ["list", "of", "variable", "names"]
        }}
        Do NOT return plain text responses, only JSON.
        '''
    else:
        system_prompt = f'''
        You are an expert data scientist analyzing data through code execution.
        The DataFrame and utilities are available in your execution context.
        You are an AI assistant that provides structured JSON responses. 

        Available objects and functions:
        - df: The input DataFrame
        - inspect_dataframe(df): Returns key DataFrame information
        - pd: pandas library
        - plt: matplotlib.pyplot

        IMPORTANT GUIDELINES:
        1. ALWAYS start by inspecting the dataframe using inspect_dataframe() and add it to "code_to_execute"
        2. After inspection, analyze the data structure before proceeding
        3. Break down complex tasks into steps
        4. Handle errors and missing data appropriately
        5. Return any created objects that need to be preserved

        Your output at each iteration must be a JSON with these fields:
        {{
            "reasoning": "Explain your thought process and what you learned",
            "next_step": "Clearly state what you will do next",
            "code_to_execute": "Your code here",
            "is_complete": false,
            "objects_to_preserve": ["list", "of", "variable", "names"]
        }}
        Do NOT return plain text responses, only JSON.
        '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_question}
    ]

    print_separator("Initial Question")
    print(f"User Question: {input_question}\n")

    is_complete = False
    max_iterations = 20
    iteration_count = 0
    last_code = None

    while not is_complete and iteration_count < max_iterations:
        iteration_count += 1
        print_separator(f"Iteration {iteration_count}")

        # Get next action from LLM
        print("\n[LLM Response]")
        response = my_completion(messages)
        print("Raw LLM Output:")
        print(response)

        # Parse the response
        try:
            action = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from string if the response isn't pure JSON
            action = extract_json_from_string(response)
            if not action:
                print("\n❌ Failed to parse JSON from LLM output")
                messages.append({
                    "role": "user", 
                    "content": "Your last response was not valid JSON. Please provide a valid JSON response."
                })
                continue

        print("\nParsed Action:")
        pretty_print_json(action)

        if not action or "code_to_execute" not in action:
            print("\n❌ Failed to get valid action from LLM")
            messages.append({
                "role": "user",
                "content": "Your response is missing the 'code_to_execute' field. Please provide valid JSON with all required fields."
            })
            continue

        if action.get("is_complete") and not action["code_to_execute"].strip():
            print("\n✅ Task completed without additional code execution")
            break
            
        # Check for loops
        if action["code_to_execute"] == last_code:
            print("\n⚠️ Detected code repetition - requesting new approach")
            messages.append({
                "role": "user",
                "content": "Warning: Detected repeated code execution. Please try a different approach."
            })
            continue

        last_code = action["code_to_execute"]

        # Execute code and capture results
        print("\n[Code Execution]")
        print("Executing code:")
        print(action["code_to_execute"])
        
        result = execute_code(action["code_to_execute"], initial_state)

        serializable_objects = {}
        if result.returned_objects:
            for key, value in result.returned_objects.items():
                if isinstance(value, pd.DataFrame):
                    # Convert DataFrame to a dict representation
                    serializable_objects[key] = {
                        "type": "DataFrame",
                        "shape": value.shape,
                        "head": value.head(5).to_dict('records'),
                        "columns": list(value.columns)
                    }
                elif isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    serializable_objects[key] = {
                        "type": "ndarray",
                        "shape": value.shape,
                        "data": value.tolist() if value.size < 1000 else "Array too large to display"
                    }
                elif isinstance(value, plt.Figure):
                    # Just store the type for matplotlib figures
                    serializable_objects[key] = {"type": "matplotlib.Figure"}
                else:
                    # For other types, try to convert to string
                    try:
                        json.dumps(value, cls=CustomJSONEncoder)  # Test if it's JSON serializable
                        serializable_objects[key] = value
                    except (TypeError, OverflowError):
                        serializable_objects[key] = str(value)
        
        # Now add the execution with serializable objects
        initial_state.add_execution({
            "reasoning": action.get("reasoning", ""),
            "next_step": action.get("next_step", ""),
            "code_to_execute": action["code_to_execute"],
            "is_complete": action.get("is_complete", False),
            "result": {
                "output": result.output,
                "error": result.error,
                "returned_objects": serializable_objects
            }
        })

        print("\nExecution Result:")
        if result.error:
            print(f"❌ Error occurred: {result.error}")
            print("Traceback:")
            print(result.traceback)
        else:
            print("✅ Execution successful")

        if result.output:
            print("\nOutput:")
            print(result.output)

        if result.returned_objects:
            print("\nReturned Objects:")
            for key, value in result.returned_objects.items():
                print(f"{key}: {str(value)[:100]}...")

        # Prepare detailed feedback for LLM
        feedback = {
            "stdout": result.output,
            "error": result.error,
            "traceback": result.traceback,
            "returned_objects": {
                k: str(v) for k, v in (result.returned_objects or {}).items()
            }
        }

        # Update state with preserved objects
        if result.returned_objects:
            objects_to_preserve = action.get("objects_to_preserve", [])
            if not objects_to_preserve and result.returned_objects.get('df') is not None:
                # Always preserve df by default if not specified
                objects_to_preserve = ['df']
                
            preserved = {
                k: v for k, v in result.returned_objects.items()
                if k in objects_to_preserve
            }
            initial_state.update(preserved)
            print("\nPreserved Objects:")
            for key in preserved:
                print(f"- {key}")

        # Update conversation
        messages.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Execution result: {json.dumps(feedback)}"}
        ])

        # Check completion
        is_complete = action.get("is_complete", False)
        if is_complete:
            print("\n✅ Task completed")

        if iteration_count >= max_iterations:
            print("\n⚠️ Maximum iterations reached")
            return "Agent reached maximum iterations without completing task."

    print_separator("Final Results")
    final_result = {
        "status": "completed" if is_complete else "terminated",
        "iterations": iteration_count,
        "execution_history": initial_state.execution_history
    }
    pretty_print_json(final_result)
    return final_result

def load_state_for_conversation(conversation_id):
    if not conversation_id:
        return AgentState({})
    try:
        with open(f"memory/{conversation_id}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return AgentState({})

def save_state_for_conversation(conversation_id, state):
    with open(f"memory/{conversation_id}.pkl", "wb") as f:
        pickle.dump(state, f)

def handle_knowledge_extraction(conversation_id, message):
    """Handle messages for the knowledge extraction agent"""
    try:
        # Initialize messages list
        messages = []

        # Fetch conversation history for context
        conversation_history = ChatHistory.query.filter_by(
            conversation_id=conversation_id,
            agent_type='knowledge_extraction'
        ).order_by(ChatHistory.timestamp).all()

        # Append previous messages to the messages list (for LLM context)
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Get your existing system prompt
        system_prompt = """
        You are a helpful agent that solves problems.
        You have access to the following tools :
        -   tool_id : current_date()
                - this function has no input parameters and returns a string as its output
                - Example tool_payload : None
        -   tool_id : run_faiss_knn(query_string : str, k : int)
                - this function has 2 input parameters query_string and k
                - Example tool_payload : {"query_string" : "What is my Name", "k" : 5}
        -   tool_id : web_search(query_string: str, num_results: int)
                - this function has 2 input parameters query_string 
                - Example tool_payload : {"query_string" : "What is my Name", "num_results" : 5}
        -   tool_id : gathered_context_visualization(query_string : str, contexts : list, distances : list)
                - this function has 3 input parameters query_string, contexts and distances
                - Example tool_payload : {"query_string" : "What is my Name", "contexts" : ["My name is this", "I have h in my name"], "distances" : [1.4,2.3]}
        -   tool_id : extract_key_terms(query_string : str)
                - this function has 1 input parameter query_string
                -- Example tool_payload : {"query_string" : "What is my Name"}
        -   tool_id: knowledge_graph(query_string: str)
                - this function has 1 input parameter query_string
                - Example tool_payload: {"query_string": "What is related to COVID-19?"}
        If there is any code, add it in the reasoning itself.
        All responses must be in the following JSON Format : 
        {
            "reasoning" : " ", # the reasoning should include all of the assitants thoughts and answer the user's questions - this is the only thing the user sees, make it conversational.
            "tool_id" : " ",
            "tool_payload" : " ",
            "is_complete" : true / false
        }
        """

        # Prepare initial context for the AI
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # Loop for processing tools (use your existing loop logic)
        not_complete = True
        final_answer = None
        reasoning_steps = []
        count = 0
        max_iterations = 10
        
        while not_complete and count < max_iterations:
            try:
                count += 1
                raw_output = my_completion(messages)
                
                try:
                    output = json.loads(raw_output)
                    if isinstance(output.get('reasoning'), str):
                        output['reasoning'] = strip_markdown(output['reasoning'])
                except json.JSONDecodeError as json_err:
                    # [error handling code remains the same]
                    continue
                
                # Handle the tool output
                tool_id = output['tool_id']
                tool_payload = output['tool_payload']
                
                if tool_id == 'run_faiss_knn':
                    tool_output = run_faiss_knn(output['tool_payload'])
                else:
                    tool_output = handle_other_tools(tool_id, output['tool_payload'])
                
                # Add AI's reasoning to the database
                ai_message = ChatHistory(
                    conversation_id=conversation_id,
                    agent_type='knowledge_extraction',
                    model_type='gpt-4o',
                    role="system",
                    tool_used=tool_id,
                    tool_payload=json.dumps(tool_payload),
                    tool_output=json.dumps(tool_output),
                    step_number=count,
                    content=output['reasoning']
                )
                
                db.session.add(ai_message)
                
                # Add spacing for better formatting in the reasoning
                formatted_reasoning = output["reasoning"].replace("\n", "\n\n")
                
                reasoning_steps.append({
                    "role": "assistant",
                    "tool_used": f"Tool being used: {output['tool_id']}",
                    "tool_payload": f"What is being searched for: {output['tool_payload']}",
                    "tool_output": tool_output,
                    "reasoning": formatted_reasoning
                })
                
                db.session.commit()
                
                if output['is_complete']:
                    # For the final answer, don't duplicate the reasoning
                    # Instead, add a clear "FINAL ANSWER:" prefix
                    final_answer = {
                        "reasoning": "FINAL ANSWER:\n\n" + formatted_reasoning,
                        "tool_used": tool_id,
                        "final_output": None  # Don't include the tool output again
                    }
                    
                    # Return only the final reasoning in the final_output
                    return json.dumps({
                        "reasoning_steps": reasoning_steps,
                        "final_output": final_answer
                    }, cls = CustomJSONEncoder)
                
                messages.append({"role": "assistant", "content": raw_output})
                if tool_output:
                    messages.append({"role": "user", "content": str(tool_output)})
                    
            except Exception as e:
                print(f"Error during processing: {e}")
                return jsonify({"error": f"An error occurred: {str(e)}"}), 500
                
        return jsonify({"error": "Processing did not complete in time."}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_model_completion(messages, model_type="gpt-4o"):
    """
    Get completion from different AI models based on model_type.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model_type: String indicating which model to use (e.g., 'gpt-4o', 'claude-sonnet')
        
    Returns:
        String containing the model's response
    """
    if model_type == "gpt-4o" or model_type.startswith("gpt-"):
        # Use your existing OpenAI function
        return my_completion(messages)
    elif model_type.startswith("claude-"):
        # Implement Claude API call here if needed
        # For now, fallback to GPT-4o
        print(f"Claude model {model_type} not implemented, falling back to GPT-4o")
        return my_completion(messages)
    else:
        # Default to GPT-4o for unknown models
        print(f"Unknown model type {model_type}, falling back to GPT-4o")
        return my_completion(messages)





@app.route('/api/conversations/<string:conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get chat history for a specific conversation"""
    try:
        messages = ChatHistory.query.filter_by(
            conversation_id=conversation_id
        ).order_by(ChatHistory.timestamp).all()
        
        # Get the agent type
        agent_type = None
        if messages:
            agent_type = messages[0].agent_type
        
        # Format the result
        result = []
        for msg in messages:
            message_data = {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }

            has_objects = False
            if hasattr(msg, 'pickled_objects') and msg.pickled_objects:
                has_objects = True
                message_data["has_pickled_objects"] = True

            if msg.code:
                message_data["code"] = msg.code
            if msg.output:
                message_data["output"] = msg.output
            if msg.error:
                message_data["error"] = msg.error
            if msg.plot_path:
                message_data["plot_path"] = msg.plot_path
            if msg.plot_paths:
                message_data["plot_paths"] = json.loads(msg.plot_paths)
            # Add tool-related fields for knowledge extraction agent
            if msg.tool_used:
                message_data["tool_used"] = msg.tool_used
                message_data["tool_payload"] = msg.tool_payload
                message_data["tool_output"] = msg.tool_output
                message_data["step_number"] = msg.step_number
            
            # Add execution data for data analysis agent if available
            if agent_type == 'data_analysis' and msg.role == 'system' and msg.content.startswith('{') and msg.content.endswith('}'):
                try:
                    # Try to parse the content as JSON (for execution history)
                    execution_data = json.loads(msg.content)
                    if 'execution_history' in execution_data:
                        # Extract the execution history
                        message_data["execution_history"] = execution_data['execution_history']
                        
                        # Extract plot paths
                        plot_paths = []
                        for execution in execution_data['execution_history']:
                            if 'result' in execution and 'returned_objects' in execution['result']:
                                if 'plot_path' in execution['result']['returned_objects']:
                                    plot_paths.append(execution['result']['returned_objects']['plot_path'])
                        
                        if plot_paths:
                            message_data["plot_paths"] = plot_paths
                except json.JSONDecodeError:
                    # If content is not valid JSON, just keep as is
                    pass
                
            result.append(message_data)
            
        return json.dumps(result, cls = CustomJSONEncoder)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()  # Convert to string
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation with the specified agent"""
    try:
        data = request.json
        agent_id = data.get('agent_id')
        
        if not agent_id:
            return jsonify({"error": "Agent ID is required"}), 400
        
        # Generate a unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Check if this is a built-in agent or custom agent
        if agent_id in ['data_analysis', 'knowledge_extraction']:
            # Handle built-in agent
            agent_type = agent_id
            model_type = 'gpt-4o'  # Default model for built-in agents
            
            # Create a welcome message
            welcome_message = ChatHistory(
                conversation_id=conversation_id,
                agent_type=agent_type,
                model_type=model_type,
                role="system",
                content=f"Welcome to the {agent_type} agent! How can I help you today?"
            )
            
            db.session.add(welcome_message)
            db.session.commit()
            
            return jsonify({
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "model_type": model_type,
                "is_built_in": True
            })
        else:
            # Handle custom agent
            definition = AgentDefinition.query.get(agent_id)
            if not definition:
                return jsonify({"error": "Agent definition not found"}), 404
                
            # Create a welcome message
            welcome_message = ChatHistory(
                conversation_id=conversation_id,
                agent_type="custom",
                model_type=definition.model_type,
                agent_definition_id=definition.id,
                role="system",
                content=f"Welcome to {definition.name}! How can I help you today?"
            )
            
            db.session.add(welcome_message)
            db.session.commit()
            
            return jsonify({
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "agent_type": "custom",
                "model_type": definition.model_type,
                "agent_name": definition.name,
                "is_built_in": False
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversation IDs and their agent types"""
    try:
        # Query distinct conversation IDs and their agent types with aggregated timestamp
        conversations = db.session.query(
            ChatHistory.conversation_id,
            ChatHistory.agent_type,
            db.func.min(ChatHistory.timestamp).label('first_timestamp'),
            db.func.max(ChatHistory.timestamp).label('latest_timestamp')
        ).group_by(ChatHistory.conversation_id, ChatHistory.agent_type).order_by(
            db.func.max(ChatHistory.timestamp).desc()
        ).all()
        
        # Format the result
        result = []
        for convo_id, agent_type, first_timestamp, latest_timestamp in conversations:
            # Fetch first user message for preview
            first_user_msg = db.session.query(ChatHistory.content).filter_by(
                conversation_id=convo_id,
                role='user'
            ).order_by(ChatHistory.timestamp.asc()).first()

            preview = (first_user_msg[0][:20] + '...') if first_user_msg else "New Chat"
            label = f"Data_Agent - {preview}" if agent_type == "data_analysis" else f"{agent_type} - {preview}"

            result.append({
                "id": convo_id,
                "agent_type": agent_type,
                "label": label,
                "latest_timestamp": latest_timestamp.isoformat()
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversations/<string:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    history = ChatHistory.query.filter_by(conversation_id=conversation_id).order_by(ChatHistory.timestamp).all()
    
    if not history:
        return jsonify({"error": "Conversation not found"}), 404

    # Get agent type
    agent_type = history[0].agent_type if history else None
    
    messages = []
    for msg in history:
        if msg.role == 'user':
            messages.append({
                "role": "user",
                "content": msg.content
            })
        else:  # assistant messages
            message_data = {
                "role": "assistant",
                "content": msg.content
            }
            
            # For knowledge extraction agent
            if msg.tool_used:
                message_data["tool_used"] = msg.tool_used
                message_data["tool_payload"] = json.loads(msg.tool_payload) if msg.tool_payload else None
                message_data["tool_output"] = json.loads(msg.tool_output) if msg.tool_output else None
                message_data["step_number"] = msg.step_number
            
            # For data analysis agent
            if agent_type == 'data_analysis' and msg.content.startswith('{') and msg.content.endswith('}'):
                try:
                    # Try to parse the content as JSON (for execution history)
                    execution_data = json.loads(msg.content)
                    if 'execution_history' in execution_data:
                        # Process execution history into a more frontend-friendly format
                        reasoning_steps = []
                        plot_paths = []
                        
                        for i, execution in enumerate(execution_data['execution_history']):
                            step = {
                                "step_number": i + 1,
                                "reasoning": execution.get('reasoning', ''),
                                "next_step": execution.get('next_step', ''),
                                "code": execution.get('code_to_execute', ''),
                            }
                            
                            # Add execution results if available
                            if 'result' in execution:
                                step["output"] = execution['result'].get('output', '')
                                step["error"] = execution['result'].get('error', '')
                                
                                # Extract any plot paths
                                if 'returned_objects' in execution['result']:
                                    returned_objects = execution['result']['returned_objects']
                                    if 'plot_path' in returned_objects:
                                        plot_path = returned_objects['plot_path']
                                        step["plot_path"] = plot_path
                                        plot_paths.append(plot_path)
                            
                            reasoning_steps.append(step)
                        
                        message_data["reasoning_steps"] = reasoning_steps
                        if plot_paths:
                            message_data["plot_paths"] = plot_paths
                except json.JSONDecodeError:
                    # If content is not valid JSON, just keep as is
                    pass
                
            messages.append(message_data)
            
    return jsonify(messages), 200
@app.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
def get_conversation_messages(conversation_id):
    try:
        messages = ChatHistory.query.filter_by(conversation_id=conversation_id).order_by(ChatHistory.timestamp.asc()).all()
        message_data = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            for msg in messages
        ]
        return jsonify(message_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/conversations/new', methods=['POST'])
def new_conversation():
    try:
        conversation_id = str(uuid.uuid4())
        return jsonify({"id": conversation_id, "preview": "New conversation"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<conversation_id>", methods=["DELETE", "OPTIONS"])
def delete_conversation(conversation_id):
    try:
        messages = ChatHistory.query.filter_by(conversation_id=conversation_id).all()
        if not messages:
            return jsonify({"error": "Conversation not found"}), 404
        ChatHistory.query.filter_by(conversation_id=conversation_id).delete()
        db.session.commit()
        return jsonify({"message": "Conversation deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/messages/<int:message_id>/objects', methods=['GET'])
def get_message_objects(message_id):
    """Get pickled objects for a specific message"""
    try:
        message = ChatHistory.query.get(message_id)
        if not message:
            return jsonify({"error": "Message not found"}), 404
            
        if not hasattr(message, 'pickled_objects') or not message.pickled_objects:
            return jsonify({"error": "No pickled objects found"}), 404
            
        # Load the objects
        objects = message.load_objects()
        
        # Convert objects to a summary format for API response
        objects_summary = {}
        for key, value in objects.items():
            if isinstance(value, pd.DataFrame):
                objects_summary[key] = {
                    "type": "DataFrame",
                    "shape": value.shape,
                    "columns": list(value.columns),
                    "head": value.head(3).to_dict('records')
                }
            elif isinstance(value, np.ndarray):
                objects_summary[key] = {
                    "type": "ndarray",
                    "shape": value.shape,
                    "sample": value.flatten()[:5].tolist() if value.size > 0 else []
                }
            else:
                objects_summary[key] = {
                    "type": type(value).__name__,
                    "summary": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                }
        
        return json.dumps({"objects": objects_summary}, cls=CustomJSONEncoder)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add a new endpoint to update the state with objects from a previous message
@app.route('/api/conversations/<string:conversation_id>/state/restore', methods=['POST'])
def restore_state_from_message(conversation_id):
    """Restore state from pickled objects in a message"""
    try:
        data = request.json
        message_id = data.get('message_id')
        
        if not message_id:
            return jsonify({"error": "Message ID is required"}), 400
            
        message = ChatHistory.query.get(message_id)
        if not message:
            return jsonify({"error": "Message not found"}), 404
            
        if not hasattr(message, 'pickled_objects') or not message.pickled_objects:
            return jsonify({"error": "No pickled objects found in this message"}), 404
            
        # Create a state message to store the restored state
        state_message = ChatHistory(
            conversation_id=conversation_id,
            agent_type=message.agent_type,
            role="system",
            content=f"State restored from message ID: {message_id}"
        )
        
        # Copy the pickled objects from the source message
        state_message.pickled_objects = message.pickled_objects
        
        db.session.add(state_message)
        db.session.commit()
        
        # Return a summary of restored objects
        objects = message.load_objects()
        restored_keys = list(objects.keys())
        
        return jsonify({
            "success": True,
            "message": f"State restored with {len(restored_keys)} objects: {', '.join(restored_keys)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os
from flask import request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    return jsonify({"file_path": filepath})
def load_file_by_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path), "df"

    elif ext == ".json":
        with open(file_path, "r") as f:
            return json.load(f), "json"

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), "text"

    elif ext == ".xlsx":
        return pd.read_excel(file_path), "df"

    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text, "pdf_text"

    return f"[Unsupported file format: {ext}]", "unsupported"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        initialize_database()
    app.run(debug=True)
    
    
    
# if file_path and os.path.exists(file_path):
#             try:
#                 content, content_type = load_file_by_type(file_path)
#                 # Ensure the content is stored under a descriptive name in preserved_objects
#                 # For a CSV, it's likely a DataFrame, so let's name it 'df_0' as in your restored objects.
#                 # You might need a more robust naming convention if handling multiple files.
#                 variable_name = "df_0" # Or dynamically generate a unique name
#                 preserved_objects[variable_name] = content

#                 # --- IMPORTANT CHANGE START ---
#                 # Inform the LLM that the data is loaded and available
#                 if isinstance(content, pd.DataFrame):
#                     preview = content.head(3).to_markdown()
#                     # Add a message to the conversation history that tells the LLM about the loaded DataFrame
#                     messages.append({"role": "system", "content": f"A DataFrame named '{variable_name}' has been loaded from the provided file. Here's a preview of its first 3 rows:\n{preview}\n\nTo proceed, inspect the DataFrame further using code like `{variable_name}.columns` or `{variable_name}.info()`."})
#                     data_loaded_and_communicated = True
#                 elif isinstance(content, dict):
#                     messages.append({"role": "system", "content": f"A dictionary named '{variable_name}' has been loaded from the provided file. Keys: {list(content.keys())}\n\nTo proceed, inspect the dictionary further using code like `print({variable_name}.keys())` or `print({variable_name}['some_key'])`."})
#                     data_loaded_and_communicated = True
#                 elif isinstance(content, str):
#                     messages.append({"role": "system", "content": f"A string named '{variable_name}' has been loaded from the provided file. Snippet:\n{content[:500]}\n\nTo proceed, inspect the string further using code like `print({variable_name}[:100])`."})
#                     data_loaded_and_communicated = True
#                 # --- IMPORTANT CHANGE END ---

#             except Exception as e:
#                 return jsonify({"error": f"Error processing uploaded file: {str(e)}"}), 500
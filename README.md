# Document Similarity Search API

## Overview

This API allows you to upload text documents, generate embeddings for them, and perform document similarity searches using the FAISS (Facebook AI Similarity Search) library. The goal is to find the most similar documents to a given query text using vector embeddings for efficient and scalable similarity search.

## Features

- *Upload Documents*: Allows uploading text documents to be added to the index.
- *Search Documents*: Search for similar documents based on a query.
- *FAISS Indexing*: Uses the FAISS library to store and retrieve document embeddings for efficient similarity search.

## Setup and Installation

### 1. Clone the repository

Clone this repository to your local machine:

```bash
git clone <repository_url>
cd <repository_folder>

2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install Dependencies

Install the required dependencies:

pip install -r requirements.txt

4. Directory Structure

The directory structure should look like this:

project/
├── data/                  # Folder for storing uploaded documents
├── index/                 # Folder for storing the FAISS index
├── src/
│   ├── api.py             # FastAPI application with endpoints
│   ├── faiss_index.py     # FAISS indexing functions
│   ├── embeddings.py      # Functions for generating embeddings for documents
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation

5. Requirements

Python 3.8+

FastAPI

FAISS

Uvicorn (for running the server)


6. Running the Server

To run the API locally, use the following command:

uvicorn src.api:app --reload

This will start the FastAPI development server. The API will be available at:

http://127.0.0.1:8000

Access it at http://127.0.0.1:8000/docs

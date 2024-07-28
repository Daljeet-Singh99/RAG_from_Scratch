
# RAG System from Scratch

This notebook demonstrates building a Retrieval Augmented Generation (RAG) system from scratch. It focuses on processing a PDF document, generating embeddings, and preparing the data for retrieval.

## Process

1. **PDF Processing:**
    - Downloads a PDF file.
    - Extracts text from each page using `PyMuPDF`.
    - Cleans and formats the extracted text.
    - Splits the text into sentences using `spaCy`.
    - Groups sentences into chunks.

2. **Embedding Generation:**
    - Uses the `all-mpnet-base-v2` model from `sentence-transformers` to generate embeddings for each text chunk.
    - Embeds chunks individually and in batches for efficiency.

3. **Data Preparation:**
    - Saves the text chunks and their corresponding embeddings to a CSV file.
    - Loads the saved data and converts embeddings to a PyTorch tensor.

## Key Components

- **Retrieval:** The system retrieves relevant text chunks based on user queries by comparing their embeddings.
- **Augmentation:** The retrieved chunks are used to augment the input to a language model for generating more informed and accurate responses.
- **Generation:** A language model generates the final output based on the augmented input.

## Requirements

- Python 3.8 or higher
- PyMuPDF
- tqdm
- sentence-transformers
- accelerate
- bitsandbytes
- flash-attn
- torch
- torchvision
- spacy

## Usage

1. Install the required libraries.
2. Run the notebook cells sequentially.
3. The processed data and embeddings will be saved for use in the RAG system.

## Note

This notebook focuses on the data processing and embedding generation steps of a RAG system. Further development is needed to implement the retrieval, augmentation, and generation components.

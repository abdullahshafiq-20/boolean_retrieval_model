# Boolean Information Retrieval System

This project implements a Boolean Information Retrieval System that supports:
- Simple term queries
- Boolean queries (using AND, OR, NOT)
- Proximity queries

---

## Technologies Used

- **Python** – Core programming language used for development.
- **NLTK** – Library for natural language processing (tokenizing, stemming, stop-word removal, etc.).
- **Streamlit** – Framework for creating an interactive web UI.

---

## Project Structure

- **boolean_retrieval_model.py** – Implements the core information retrieval logic.
- **main.py** – Streamlit-based web UI for user interaction.
- **requirements.txt** – Lists dependencies needed to run the project.
- **Abstracts/** – Contains text documents to be indexed.
- **stop_words.txt** – Stores common stop words used in text processing.
- **indexes.json** – Stores precomputed indexes for efficient retrieval.

---

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Console Version:**
   ```bash
   python boolean_retrieval_model.py
   ```

3. **Run the Web UI:**
   ```bash
   streamlit run main.py
   ```

---

## Query Examples

- **Simple Term Query:**
  ```
  computer
  ```
- **Boolean Query:**
  ```
  computer AND science NOT data
  ```
- **Proximity Query:**
  ```
  computer science /5
  ```
  *(Finds occurrences where 'computer' and 'science' appear within 5 words of each other.)*

---

## Application Links

- **Deployed App:** [Access the Web App](https://brm-4489.streamlit.app/)

---


## Additional Information

- **Indexing:** Processed documents are stored in `indexes.json` for faster access.
- **Robust Processing:** Handles different file encodings to ensure smooth document retrieval.
- **Extensibility:** Easily adaptable for additional query types and retrieval functionalities.

This documentation provides an overview of the Boolean Information Retrieval System. For further details, refer to the source code comments and inline documentation.

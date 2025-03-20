import streamlit as st
from boolean_retrieval_model import InformationRetrievalSystem
import os

class IRSystemWebUI:
    def __init__(self):
        self.ir_system = InformationRetrievalSystem("Abstracts", "stop_words.txt")
        self.setup_ui()

    def setup_ui(self):
        # Set page config
        st.set_page_config(
            page_title="Information Retrieval System",
            page_icon="🔍",
            layout="wide"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stSelectbox {
            border-radius: 6px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("Information Retrieval System")
        st.markdown("---")

        # Search Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query_type = st.selectbox(
                "Query Type",
                [
                    "Simple Term Query",
                    "Boolean Query (AND, OR, NOT)",
                    "Proximity Query (/k)"
                ]
            )
            
            query = st.text_input("Enter your query...", key="query")

        with col2:
            st.markdown("### Query Syntax Guide")
            st.info("""
            📝 **Simple Term Query:**
            - Enter a single word (e.g., "computer")
            
            🔍 **Boolean Query:**
            - Use AND, OR, NOT operators
            - Example: "computer AND science NOT data"
            
            🎯 **Proximity Query:**
            - Use two terms and /k format
            - Example: "computer science /5"
            """)

        # Search Button
        if st.button("Search", type="primary"):
            if not query:
                st.warning("Please enter a query")
            else:
                try:
                    # Process query and get results
                    doc_ids = self.ir_system.process_query(query)
                    
                    # Display results
                    if doc_ids:
                        st.success(f"Found {len(doc_ids)} matching documents")
                        
                        # Create an expander for results
                        with st.expander("View Results", expanded=True):
                            st.markdown("### Matching Documents", )
                            
                            # Create a nice looking table for results
                            results_data = []
                            for doc_id in doc_ids:
                                filename = self.ir_system.doc_ids[doc_id]
                                results_data.append({
                                    "Document ID": doc_id,
                                    "Filename": filename
                                })
                            
                            st.table(results_data)
                            
                    else:
                        st.warning("No matching documents found. Try modifying your search terms.")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

def main():
    IRSystemWebUI()

if __name__ == "__main__":
    main()
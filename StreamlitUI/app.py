

import streamlit as st
import sys
import os
import time

# Set page config first
st.set_page_config(
    page_title="Legal Discovery & RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the retrieval logic
try:
    from RAG.retrieve_hybrid import get_hybrid_results
except ImportError as e:
    st.error(f"Error importing RAG module: {e}. Please ensure you are running this from the project root.")
    st.stop()

# --- Custom CSS for Professional Legal Tech UI ---
st.markdown("""
<style>
    /* Global Styles */
    .reportview-container {
        background: #F0F2F6;
    }
    .main {
        background-color: #FAFAFA;
        padding-top: 2rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
    }
    
    /* Search Bar Styling */
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stTextInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 4px solid #2563eb;
        transition: box-shadow 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    
    .source-tag {
        background: #eff6ff;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #bfdbfe;
    }

    .para-tag {
        background: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        margin-left: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .score-text {
        color: #64748b;
        font-weight: 500;
    }
    
    .content-text {
        font-family: 'Georgia', serif; /* Serif for better reading of legal text */
        font-size: 1rem;
        line-height: 1.6;
        color: #334155;
    }
    
    /* Button Styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration REMOVED ---
# Moving settings to an expander for a cleaner look

# --- Main Interface ---
st.title("⚖️ Legal Intelligence Search")
st.markdown("Ask questions or search for specific legal concepts within your document knowledge base.")

# Search Input Section
search_container = st.container()
with search_container:
    col_search, col_btn = st.columns([6, 1])
    
    with col_search:
        query = st.text_input(
            "Search Query", 
            placeholder="e.g. 'What are the penalties for breach of contract?'", 
            label_visibility="collapsed"
        )
    
    with col_btn:
        # Align button with input
        st.markdown('<div style="margin-top: 3px;"></div>', unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)

# Settings in Expander (Hidden by default)
with st.expander("⚙️ Search Configuration"):
    col_k, col_alpha = st.columns(2)
    with col_k:
        k_value = st.slider(
            "Documents to Retrieve (k)", 
            min_value=1, 
            max_value=20, 
            value=5,
            step=1,
            help="Higher values retrieve more context but may reduce precision."
        )
    with col_alpha:
        alpha_value = st.slider(
            "Hybrid Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0.0 = Keyword Only, 1.0 = Semantic Only."
        )

# --- Logic & Display ---
if query:
    st.markdown("### Search Results")
    
    with st.spinner("Analyzing documents..."):
        # Artificial delay for UI feel if response is too instant, or remove if not needed
        # time.sleep(0.3) 
        try:
            results = get_hybrid_results(query, top_k=k_value, alpha=alpha_value)
            
            if not results:
                st.warning("No relevant documents found. Try adjusting the keywords or Alpha value.")
            else:
                for i, m in enumerate(results, 1):
                    meta = m.get('metadata', {})
                    source = meta.get('source', 'Unknown Document')
                    para_id = meta.get('para_id', '#')
                    text = meta.get('text', '')
                    score = m.get('score', 0.0)
                    
                    # Modern Card Design using Container
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="meta-row">
                                <div>
                                    <span class="source-tag">📄 {source}</span>
                                    <span class="para-tag">Paragraph {para_id}</span>
                                </div>
                                <span class="score-text">Relevance: {score:.2%}</span>
                            </div>
                            <div class="content-text">
                                {text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Search failed: {str(e)}")
else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #94a3b8;">
        <h4>Ready to Search</h4>
        <p>Enter a query above to explore the legal database.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa; font-size: 0.8rem;'>Powered by Hybrid RAG (BM25 + Semantic)</div>", unsafe_allow_html=True)

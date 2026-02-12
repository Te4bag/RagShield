import streamlit as st
import os
import shutil
from pathlib import Path
from ingest import DocumentLoader
from index import chunk_documents, RagShieldIndex
from rag import Retriever, RagGenerator
from verify import NLIAuditor

# 1. Configuration & Enhanced Styling
st.set_page_config(
    page_title="RagShield - AI Document Auditor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with inline verification
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styling */
    .main {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Header */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8b8b9a;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar enhancements */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #252538 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #667eea;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        background: rgba(30, 30, 47, 0.6);
        color: #ffffff;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 30, 47, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px dashed rgba(102, 126, 234, 0.3);
    }
    
    /* Response container */
    .response-container {
        background: rgba(30, 30, 47, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        line-height: 1.8;
        font-size: 1.05rem;
        color: #e5e5e5;
    }
    
    /* Inline verification - sentences with colored underlines */
    .sentence-verified {
        position: relative;
        cursor: help;
        border-bottom: 3px solid rgba(16, 185, 129, 0.6);
        transition: all 0.2s ease;
        padding-bottom: 2px;
    }
    
    .sentence-verified:hover {
        background: rgba(16, 185, 129, 0.1);
        border-bottom-color: #10b981;
    }
    
    .sentence-contradiction {
        position: relative;
        cursor: help;
        border-bottom: 3px solid rgba(239, 68, 68, 0.6);
        transition: all 0.2s ease;
        padding-bottom: 2px;
    }
    
    .sentence-contradiction:hover {
        background: rgba(239, 68, 68, 0.1);
        border-bottom-color: #ef4444;
    }
    
    .sentence-neutral {
        position: relative;
        cursor: help;
        border-bottom: 3px solid rgba(251, 191, 36, 0.6);
        transition: all 0.2s ease;
        padding-bottom: 2px;
    }
    
    .sentence-neutral:hover {
        background: rgba(251, 191, 36, 0.1);
        border-bottom-color: #fbbf24;
    }
    
    /* Tooltip on hover */
    .sentence-verified::after,
    .sentence-contradiction::after,
    .sentence-neutral::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%) translateY(-8px);
        background: rgba(0, 0, 0, 0.95);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease, transform 0.2s ease;
        z-index: 1000;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .sentence-verified:hover::after,
    .sentence-contradiction:hover::after,
    .sentence-neutral:hover::after {
        opacity: 1;
        transform: translateX(-50%) translateY(-12px);
    }
    
    /* Stats badges */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .stat-badge {
        background: rgba(30, 30, 47, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        display: block;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #8b8b9a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Legend */
    .legend-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        font-size: 0.9rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-line {
        width: 40px;
        height: 3px;
        border-radius: 2px;
    }
    
    .legend-verified { background: #10b981; }
    .legend-contradiction { background: #ef4444; }
    .legend-neutral { background: #fbbf24; }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 47, 0.6);
        border-radius: 8px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Info alert for instructions */
    .instruction-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #b8bcc7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'indexed' not in st.session_state:
    st.session_state.indexed = False

# Header with better branding
st.markdown("# 🛡️ RagShield")
st.markdown('<p class="subtitle">AI-Powered Document Verification & Fact-Checking System</p>', unsafe_allow_html=True)

UPLOAD_DIR = Path("demo/example_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 2. Simplified Sidebar
with st.sidebar:
    st.markdown("##  Document Management")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF documents to build your knowledge base"
    )
    
    # Auto-save uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            # Only write if file doesn't exist or is different
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.indexed = False  # Mark as needs re-indexing
    
    if st.button("🗑️ Clear All Documents", use_container_width=True):
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir()
        index = RagShieldIndex()
        try:
            index.client.delete_collection("rag_shield_docs")
        except:
            pass
        st.session_state.indexed = False
        st.success(" System Cleared")
        st.rerun()

    st.markdown("---")
    
    # Show uploaded documents
    if os.path.exists(UPLOAD_DIR):
        docs = list(UPLOAD_DIR.glob("*.pdf"))
        if docs:
            st.markdown("### 📄 Current Documents")
            for doc in docs:
                size_mb = doc.stat().st_size / (1024 * 1024)
                st.markdown(f"- `{doc.name}` ({size_mb:.1f}MB)")
        else:
            st.info(" No documents uploaded yet")
    

# 3. Main Query Interface
st.markdown("### 💬 Ask a Question")

query = st.text_area(
    "",
    placeholder="Enter your query about the uploaded documents...",
    height=100,
    label_visibility="collapsed"
)

if st.button("🔍 Analyze Query", use_container_width=True, type="primary"):
    if not query:
        st.warning(" Please enter a query")
    else:
        # Check if we need to index
        docs = list(UPLOAD_DIR.glob("*.pdf")) if os.path.exists(UPLOAD_DIR) else []
        
        if not docs:
            st.error(" No documents found. Please upload PDF documents first.")
        else:
            # Auto-index if needed
            if not st.session_state.indexed:
                with st.spinner(" Indexing documents..."):
                    loader = DocumentLoader(str(UPLOAD_DIR))
                    index = RagShieldIndex()
                    try:
                        index.client.delete_collection("rag_shield_docs")
                    except:
                        pass
                    index.collection = index.client.get_or_create_collection(
                        name="rag_shield_docs", embedding_function=index.embedding_fn
                    )
                    index.add_documents(chunk_documents(loader.load()))
                    st.session_state.indexed = True
            
            with st.spinner("🛡️ Analyzing and verifying response..."):
                retriever = Retriever()
                context, metadata = retriever.get_context(query)
                
                if not context:
                    st.error(" No relevant context found in the documents.")
                else:
                    generator = RagGenerator()
                    response = generator.generate_answer(query, context)
                    
                    auditor = NLIAuditor()
                    audit_results = auditor.audit_response(response, context)
                    
                    # Statistics Summary
                    st.markdown("---")
                    st.markdown("###  Audit Summary")
                    
                    entailment_count = sum(1 for r in audit_results if r['verdict'] == 'ENTAILMENT')
                    contradiction_count = sum(1 for r in audit_results if r['verdict'] == 'CONTRADICTION')
                    neutral_count = sum(1 for r in audit_results if r['verdict'] == 'NEUTRAL')
                    total = len(audit_results)
                    
                    # Stats badges
                    stats_html = f"""
                    <div class="stats-container">
                        <div class="stat-badge">
                            <span class="stat-value" style="color: #10b981;">{entailment_count}</span>
                            <span class="stat-label">✅ Verified</span>
                        </div>
                        <div class="stat-badge">
                            <span class="stat-value" style="color: #ef4444;">{contradiction_count}</span>
                            <span class="stat-label">❌ Contradictions</span>
                        </div>
                        <div class="stat-badge">
                            <span class="stat-value" style="color: #fbbf24;">{neutral_count}</span>
                            <span class="stat-label">⚠️ Neutral</span>
                        </div>
                        <div class="stat-badge">
                            <span class="stat-value">{total}</span>
                            <span class="stat-label"> Total Claims</span>
                        </div>
                    </div>
                    """
                    st.markdown(stats_html, unsafe_allow_html=True)
                    
                    # 4. Response with inline verification
                    st.markdown("### 💬 Response (Hover for verification)")
                    
                    # Legend
                    legend_html = """
                    <div class="legend-container">
                        <div class="legend-item">
                            <div class="legend-line legend-verified"></div>
                            <span>Verified</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line legend-contradiction"></div>
                            <span>Contradicted</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-line legend-neutral"></div>
                            <span>Neutral</span>
                        </div>
                    </div>
                    """
                    st.markdown(legend_html, unsafe_allow_html=True)
                    
                    # Build response with inline verification
                    response_html = '<div class="response-container">'
                    
                    for idx, res in enumerate(audit_results):
                        verdict = res['verdict']
                        confidence = res['confidence']
                        sentence = res['sentence']
                        
                        # Determine class and tooltip
                        if verdict == 'ENTAILMENT':
                            css_class = "sentence-verified"
                            icon = "✅"
                        elif verdict == 'CONTRADICTION':
                            css_class = "sentence-contradiction"
                            icon = "❌"
                        else:
                            css_class = "sentence-neutral"
                            icon = "⚠️"
                        
                        # Create tooltip text
                        tooltip = f"{icon} {verdict} | Confidence: {confidence:.2f}"
                        
                        # Wrap sentence in span with tooltip
                        response_html += f'<span class="{css_class}" data-tooltip="{tooltip}">{sentence}</span> '
                    
                    response_html += '</div>'
                    st.markdown(response_html, unsafe_allow_html=True)
                    
                    # Detailed analysis in expander
                    with st.expander("🔍 View Detailed Analysis", expanded=False):
                        for idx, res in enumerate(audit_results, 1):
                            verdict = res['verdict']
                            confidence = res['confidence']
                            
                            if verdict == 'ENTAILMENT':
                                icon = "✅"
                                color = "#10b981"
                            elif verdict == 'CONTRADICTION':
                                icon = "❌"
                                color = "#ef4444"
                            else:
                                icon = "⚠️"
                                color = "#fbbf24"
                            
                            st.markdown(f"""
                            **{idx}. {icon} {res['sentence']}**
                            - Verdict: <span style="color: {color}; font-weight: 600;">{verdict}</span>
                            - Confidence: {confidence:.2f}
                            """, unsafe_allow_html=True)
                            st.markdown("---")

                    # Sources
                    with st.expander(" View Source Documents"):
                        unique_sources = list(set([m['doc_id'] for m in metadata]))
                        st.markdown("**Documents used for this analysis:**")
                        for source in unique_sources:
                            st.markdown(f"- `{source}`")
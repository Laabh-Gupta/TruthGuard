# frontend/app.py

import os
import sys
import tempfile
os.environ["ANONYMIZED_TELEMETRY"] = "False"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ingestion.pdf_loader import load_pdf, chunk_pages
from retrieval.vector_store import VectorStore
from retrieval.rag_pipeline import RAGPipeline
from evaluation.ragas_evaluator import RAGASEvaluator

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TruthGuard",
    page_icon="🛡️",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ TruthGuard")
st.caption("AI Answer Verification — Know what's grounded, what's not.")
st.divider()

# ── Session state ─────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "evaluator" not in st.session_state:
    st.session_state.evaluator = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF — research paper, report, contract, resume"
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Ingest
                pages = load_pdf(tmp_path)
                chunks = chunk_pages(pages)

                # Store
                store = VectorStore()
                store.add_chunks(chunks)

                # Init pipeline
                st.session_state.pipeline = RAGPipeline()
                st.session_state.evaluator = RAGASEvaluator()
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded_file.name

                os.unlink(tmp_path)

            st.success(f"✅ Loaded: {uploaded_file.name}")
            st.info(f"📦 {len(chunks)} chunks indexed")

    st.divider()
    st.header("⚙️ Settings")

    run_ragas = st.toggle(
        "Run RAGAS Evaluation",
        value=False,
        help="Slower but gives faithfulness + precision scores"
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="More chunks = more context but slower"
    )

    st.divider()
    st.markdown("**Models**")
    st.markdown("🤖 RAG: `phi3:mini`")
    st.markdown("⚡ Eval: `gemma3:1b`")
    st.markdown("🔢 Embed: `BAAI/bge-small-en-v1.5`")

# ── Main area ─────────────────────────────────────────────────
if not st.session_state.pdf_loaded:
    # Empty state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload a PDF in the sidebar")
    with col2:
        st.info("**Step 2**\n\nAsk any question about it")
    with col3:
        st.info("**Step 3**\n\nSee what's grounded vs hallucinated")

else:
    # Question input
    question = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What are the main findings of this paper?",
        key="question_input"
    )

    ask_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if ask_button and question:
        with st.spinner("Running TruthGuard analysis..."):
            result = st.session_state.pipeline.query(question, top_k=top_k)

        report = result["hallucination_report"]
        confidence = result["confidence"]

        # ── Confidence banner ──────────────────────────────
        st.divider()
        trust = confidence["trust_level"]
        score = confidence["confidence_pct"]
        icon = confidence["trust_icon"]

        if trust == "HIGH":
            banner = st.success
        elif trust == "MEDIUM":
            banner = st.warning
        else:
            banner = st.error

        banner(
            f"{icon} **{trust} CONFIDENCE** — {score} grounded   |   "
            f"✅ {report['supported_count']} supported   "
            f"⚠️ {report['weak_count']} weak   "
            f"❌ {report['unsupported_count']} unsupported"
        )

        # ── Two column layout ──────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            # Cited answer
            st.subheader("📄 Answer")
            st.markdown(result["cited_answer"])

            # Sentence breakdown
            st.subheader("🔍 Sentence Analysis")
            for i, s in enumerate(report["sentences"]):
                if s["verdict"] == "SUPPORTED":
                    color = "green"
                    icon = "✅"
                elif s["verdict"] == "WEAK":
                    color = "orange"
                    icon = "⚠️"
                else:
                    color = "red"
                    icon = "❌"

                with st.expander(
                    f"{icon} Sentence {i+1} — "
                    f"{s['verdict']} (score: {s['best_score']})"
                ):
                    st.markdown(f"**Sentence:**\n{s['sentence']}")
                    st.markdown(f"**Best matching evidence:**")
                    st.caption(s["best_chunk_text"])
                    st.markdown(
                        f"Source: `{s['best_chunk_source']}` "
                        f"— Page {s['best_chunk_page']}"
                    )

            # Hallucination alert
            if report["unsupported_sentences"]:
                st.subheader("🚨 Potential Hallucinations")
                for s in report["unsupported_sentences"]:
                    st.error(
                        f"❌ **Unsupported claim detected**\n\n"
                        f"{s['sentence']}\n\n"
                        f"Similarity to any source: {s['best_score']}"
                    )

        with col_right:
            # Confidence breakdown
            st.subheader("📊 TruthGuard Scores")

            st.metric("Confidence Score", score)
            st.metric("Grounding Score",
                      f"{round(confidence['grounding_score']*100, 1)}%")
            st.metric("Avg Retrieval",
                      f"{round(confidence['avg_retrieval_score']*100, 1)}%")

            st.divider()
            st.markdown(f"💡 *{confidence['recommendation']}*")

            # RAGAS scores
            if run_ragas:
                st.divider()
                st.subheader("📐 RAGAS Evaluation")
                with st.spinner("Running RAGAS... (~2 min)"):
                    ragas_result = st.session_state.evaluator.evaluate_single(
                        question=question,
                        answer=result["answer"],
                        contexts=[c["text"] for c in result["sources"]]
                    )

                if ragas_result["ragas_available"]:
                    st.metric("Faithfulness",
                              ragas_result["faithfulness"] or "N/A")
                    st.metric("Answer Relevancy",
                              ragas_result["answer_relevancy"] or "N/A")
                    st.metric("Context Precision",
                              ragas_result["context_precision"] or "N/A")
                else:
                    st.warning("RAGAS evaluation unavailable")

            # Sources
            st.divider()
            st.subheader("📚 Sources Used")
            for i, src in enumerate(result["sources"]):
                st.markdown(
                    f"**[{i+1}]** `{src['source']}` — "
                    f"Page {src['page']} "
                    f"*(sim: {src['similarity']})*"
                )
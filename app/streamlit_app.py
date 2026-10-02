"""Streamlit UI for Job Application Assistant."""

import os

import streamlit as st
from ui_helpers import extract_text_from_bytes, run_analysis

from jobassist.llm import get_llm_client
from jobassist.tiers import TaskRequirements, load_tiers_config, select_tier

# -- Page Config & CSS --
st.set_page_config(page_title="Job Application Assistant", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    /* Soft card backgrounds and subtle shadows */
    .stFileUploader, .stTextArea {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        background-color: #F5F7FB;
        color: #4F46E5;
        border: 1px solid #4F46E5;
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -- Helper Functions --
def get_file_content(uploaded_file, text_input) -> str:
    """Get text either from file upload or direct text input."""
    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error(f"File {uploaded_file.name} exceeds 5 MB limit.")
            return ""
        try:
            return extract_text_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            return ""
    return text_input.strip()

def count_tokens_fast(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text.split()) * 1.3)

def has_api_env_vars() -> bool:
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"))

# -- State Init --
if "results" not in st.session_state:
    st.session_state.results = None
if "answer" not in st.session_state:
    st.session_state.answer = None

# -- Header --
st.title("📄 Job Application Assistant")
st.markdown("Upload a resume and job description to get a validated, structured assessment.")

# -- Sidebar --
with st.sidebar:
    st.header("Settings")
    
    api_available = has_api_env_vars()
    backend_options = ["MockLLM (Offline)", "API Backend"]
    if not api_available:
        st.warning("API Backend disabled. Set LLM_API_KEY to enable.")
        backend_idx = 0
    else:
        backend_idx = 1
        
    backend_choice = st.radio("LLM Backend", options=backend_options, index=backend_idx, disabled=not api_available)
    backend_str = "api" if "API" in backend_choice else "mock"
    
    tiers = load_tiers_config()
    tier_names = list(tiers.keys())
    
    # Auto-recommend tier logic based on typical single assessment
    req = TaskRequirements(
        task_type="complex_technical_feedback", 
        is_bulk=False,
        requires_complex_reasoning=True,
        requires_strict_schema=True,
        expected_context_size=3000
    )
    recommended_tier, _ = select_tier(req, tiers)
    default_tier_idx = tier_names.index(recommended_tier.name.lower().replace("tier-", "")) if recommended_tier.name.lower().replace("tier-", "") in tier_names else tier_names.index("large")
    
    selected_tier_name = st.selectbox("Model Tier", options=tier_names, index=default_tier_idx)
    selected_tier = tiers[selected_tier_name]
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    show_cost = st.checkbox("Show cost estimate", value=True)
    run_hallucination = st.checkbox("Run hallucination check", value=True)

# -- Main Area --
col1, col2 = st.columns(2)

resume_text = ""
with col1:
    st.subheader("Resume")
    res_tab1, res_tab2 = st.tabs(["Upload File", "Paste Text"])
    with res_tab1:
        res_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    with res_tab2:
        res_text_input = st.text_area("Or paste resume text here", height=150)
        
    resume_text = get_file_content(res_file, res_text_input)
    if resume_text:
        st.caption(f"Loaded: {len(resume_text)} characters | ~{count_tokens_fast(resume_text)} tokens")
        with st.expander("Preview extracted text"):
            st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))

jd_text = ""
with col2:
    st.subheader("Job Description")
    jd_tab1, jd_tab2 = st.tabs(["Upload File", "Paste Text"])
    with jd_tab1:
        jd_file = st.file_uploader("Upload JD (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    with jd_tab2:
        jd_text_input = st.text_area("Or paste JD text here", height=150)
        
    jd_text = get_file_content(jd_file, jd_text_input)
    if jd_text:
        st.caption(f"Loaded: {len(jd_text)} characters | ~{count_tokens_fast(jd_text)} tokens")
        with st.expander("Preview extracted text"):
            st.text(jd_text[:1000] + ("..." if len(jd_text) > 1000 else ""))

# Additional Documents
extra_docs_text = ""
with st.expander("Additional documents (Optional)"):
    extra_files = st.file_uploader("Upload cover letter, portfolio, etc.", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if extra_files:
        docs = []
        for f in extra_files:
            text = get_file_content(f, "")
            if text:
                docs.append(f"--- Document: {f.name} ---\n{text}")
        extra_docs_text = "\n\n".join(docs)
        if extra_docs_text:
            st.success(f"Loaded {len(extra_files)} additional documents.")

question_input = st.text_input("Ask a question about this candidate (optional)")

# -- Action --
col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_clicked = st.button("Analyze candidate", type="primary", disabled=not (resume_text and jd_text))
with col_btn2:
    if st.button("Reset"):
        st.session_state.results = None
        st.session_state.answer = None
        st.rerun()

# -- Execution --
if analyze_clicked:
    st.session_state.results = None
    st.session_state.answer = None
    
    with st.spinner("Analyzing candidate..."):
        try:
            res = run_analysis(
                resume_text=resume_text,
                jd_text=jd_text,
                backend=backend_str,
                tier=selected_tier,
                extra_docs=extra_docs_text,
                temperature=temperature
            )
            st.session_state.results = res
            
            if question_input:
                llm = get_llm_client(backend=backend_str)
                from jobassist.prompts import build_question_prompt
                prompt = build_question_prompt(jd_text, resume_text + "\n" + extra_docs_text, question_input)
                st.session_state.answer = llm.generate(prompt, temperature=temperature)
                
        except Exception as e:
            st.error(f"Analysis failed: {e!s}")

# -- Results Display --
if st.session_state.answer:
    st.info(f"**Answer to your question:**\n\n{st.session_state.answer}")

if st.session_state.results:
    res = st.session_state.results
    
    if not res.get("success"):
        st.error(f"Validation Error: {res.get('error')}")
        st.json(res.get("validation_result", {}))
    else:
        assessment = res["assessment"]
        
        tab_assess, tab_ground, tab_json, tab_cost = st.tabs([
            "Assessment", "Evidence & Grounding", "Raw JSON", "Cost & Tokens"
        ])
        
        with tab_assess:
            st.header(assessment.candidate_name)
            
            # Score and Recommendation
            col_score, col_rec, col_exp = st.columns(3)
            with col_score:
                st.metric("Score", f"{assessment.score}/100")
                st.progress(assessment.score / 100.0)
            with col_rec:
                rec = assessment.recommendation
                color = "green" if "Yes" in rec else "red" if "No" in rec else "orange"
                st.markdown(f"**Recommendation**<br><span style='color:{color}; font-size:1.5rem; font-weight:bold;'>{rec}</span>", unsafe_allow_html=True)
            with col_exp:
                st.metric("Experience", f"{assessment.experience_years} years")
                
            st.markdown("**Skills**")
            skills_html = "".join([f"<span class='pill'>{s}</span>" for s in assessment.skills])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            st.markdown("**Reasoning**")
            st.write(assessment.reasoning)
            
            st.markdown("**Requirement Breakdown**")
            req_data = []
            for r in assessment.requirement_assessments:
                req_data.append({
                    "Requirement": r.requirement,
                    "Status": r.status,
                    "Evidence": r.evidence
                })
            st.dataframe(req_data, use_container_width=True)

        with tab_ground:
            st.subheader("Global Evidence Quotes")
            for ev in assessment.evidence:
                st.markdown(f"- \"*{ev}*\"")
                
            st.subheader("Missing Information")
            if assessment.missing_information:
                for m in assessment.missing_information:
                    st.markdown(f"- {m}")
            else:
                st.write("None.")
                
            if run_hallucination:
                st.subheader("Grounding Report")
                report = res["grounding_report"]
                if report.passed:
                    st.success("✅ All claims supported by the document text.")
                else:
                    st.error("❌ Hallucination / Grounding Errors Detected:")
                    for uc in report.unsupported_claims:
                        st.markdown(f"- **Unsupported Claim:** {uc}")
                    for me in report.missing_evidence:
                        st.markdown(f"- **Missing Evidence:** {me}")

        with tab_json:
            st.download_button("Download JSON", data=res["raw_json"], file_name="assessment.json", mime="application/json")
            st.json(res["raw_json"])

        with tab_cost:
            cost_info = res["cost_info"]
            st.metric("Input Tokens", cost_info["input_tokens"])
            st.metric("Output Tokens", cost_info["output_tokens"])
            
            if show_cost:
                st.markdown("### Estimated Cost")
                st.write(f"**Tier:** {selected_tier.name}")
                st.write(f"**Uncached Cost:** ${cost_info['cost_no_cache']:.6f}")
                st.write(f"**Cached Cost:** ${cost_info['cost_cache']:.6f}")

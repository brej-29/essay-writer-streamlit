<div align="center">
  <h1>📝 Essay Writer App</h1>
  <p><i>Research-assisted, multi-draft essay generator with live streaming, exports (MD/TXT/DOCX/PDF), ZIP bundles, and LangSmith feedback — built with Streamlit + LangGraph + LangChain</i></p>
</div>

<br>

<div align="center">
  <a href="https://essay-writer-app.streamlit.app/">
    <img alt="Live App" src="https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  </a>
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b">
  <img alt="AI Stack" src="https://img.shields.io/badge/AI%20Stack-LangChain%20%7C%20LangGraph-orange">
  <img alt="Observability" src="https://img.shields.io/badge/Observability-LangSmith-purple">
  <img alt="Research" src="https://img.shields.io/badge/Web%20Research-Tavily-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/License-Check%20Repo-black">
</div>

<div align="center">
  <br>
  <b>Built with the tools and technologies:</b>
  <br><br>
  <code>Python</code> | <code>Streamlit</code> | <code>LangChain</code> | <code>LangGraph</code> | <code>OpenAI</code> | <code>Tavily</code> | <code>LangSmith</code> | <code>python-docx</code> | <code>ReportLab</code>
</div>

---

## **Live App**
https://essay-writer-app.streamlit.app/

---

## **Screenshot**

<img width="1778" height="982" alt="image" src="https://github.com/user-attachments/assets/a2aaca8b-0bb0-4f7a-a9ce-6b2ca98b989c" />

<img width="1918" height="923" alt="image" src="https://github.com/user-attachments/assets/7d509745-fdd6-4648-83ba-e8673672e614" />


---

## **Table of Contents**
* [Overview](#overview)
* [Features](#features)
* [Getting Started](#getting-started)
    * [Project Structure](#project-structure)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Configuration](#configuration)
    * [Usage](#usage)
* [How It Works](#how-it-works)
* [Deployment](#deployment)
* [Troubleshooting](#troubleshooting)
* [License](#license)
* [Contact](#contact)

---

## **Overview**

Essay Writer App is an interactive Streamlit application that generates high-quality essays using a structured, multi-step agent workflow:

- Plan an outline
- (Optional) Research the topic using web search
- Draft an essay
- Critique the draft
- (Optional) Research to address critique
- Revise in a loop until `max_revisions` is reached

The UI provides live streaming updates, version history (drafts + critiques), multi-format exports, and a one-click ZIP bundle containing the complete run. Each run is traceable in LangSmith, and users can submit 👍/👎 feedback directly into LangSmith for continuous improvement.

---

## **Features**

- **Modern Streamlit UI**
  - Sidebar controls for model, temperature, tone, audience, length, paragraphs
  - Toggle for web research (Tavily)
  - Max revisions for iterative refinement
  - Live progress indicators and collapsible live panels

- **Agentic Essay Workflow (LangGraph)**
  - Planner → research → writer → critique → research-for-critique → revision loop

- **Web Research (Optional)**
  - Uses Tavily search to gather concise notes to improve factual grounding

- **Live Streaming UX**
  - Per-node collapsibles (Plan / Research / Draft / Critique)
  - Research notes show ONLY new additions (no re-printing duplicates)
  - Caps displayed notes/versions to prevent UI flooding

- **Run History (In-Session)**
  - Stores multiple runs per session
  - Load/compare older runs from the sidebar

- **Exports**
  - Download final as **Markdown**, **TXT**, **DOCX**, **PDF**
  - Download a **ZIP bundle** containing plan + notes + drafts + critiques + final + metadata + exports

- **LangSmith Observability + Feedback**
  - Automatic tracing for each run
  - “Was this helpful?” 👍/👎 + comment logged as LangSmith feedback

---

## **Getting Started**

Follow these steps to set up and run the project locally.

### **Project Structure**

    essay-writer-streamlit/
    ├─ app.py
    ├─ core/
    │  ├─ __init__.py
    │  ├─ config.py              # Streamlit secrets loader + missing-secret errors
    │  ├─ telemetry.py           # LangSmith env configuration helper
    │  ├─ prompts.py             # Planner/Writer/Critique prompts
    │  ├─ schemas.py             # Pydantic configs + AgentState typings
    │  ├─ research.py            # Tavily wrapper + caching + formatting
    │  ├─ graph.py               # LangGraph pipeline + streaming runner
    │  ├─ feedback.py            # LangSmith feedback submit helper
    │  ├─ exporters.py           # MD/TXT/DOCX/PDF exporters (in-memory)
    │  └─ bundle_zip.py          # Full-run ZIP bundler (includes docx/pdf if present)
    ├─ .streamlit/
    │  └─ secrets.toml           # local-only secrets (DO NOT COMMIT)
    ├─ requirements.txt
    └─ README.md

### **Prerequisites**
- Python **3.10+** recommended (3.11 is ideal)
- API keys:
  - **OpenAI** (for LLM generation)
  - **Tavily** (optional, for web research)
  - **LangSmith** (optional but recommended, for tracing + feedback)

### **Installation**
1) Create and activate a virtual environment (optional but recommended).

    Windows (PowerShell):
    - python -m venv .venv
    - .\.venv\Scripts\Activate.ps1

    macOS/Linux:
    - python3 -m venv .venv
    - source .venv/bin/activate

2) Install dependencies.

    - pip install -r requirements.txt

### **Configuration**
This project uses **Streamlit Secrets** (no .env).

1) Create `.streamlit/secrets.toml` (local only):

    OPENAI_API_KEY = "your-openai-key"
    TAVILY_API_KEY = "your-tavily-key"

    LANGCHAIN_API_KEY = "your-langsmith-key"
    LANGCHAIN_TRACING_V2 = "true"
    LANGCHAIN_PROJECT = "essay-writer-streamlit"

2) Ensure `.streamlit/secrets.toml` is NOT committed (it should be ignored via `.gitignore`).

### **Usage**
Run the app locally:

    streamlit run app.py

In the app:
1) Enter your essay topic/task
2) Choose tone/audience/length and optional research
3) Click **Generate Essay**
4) Review:
   - Outline
   - Research notes
   - Draft versions
   - Critiques
   - Final + exports + ZIP bundle
5) Submit 👍/👎 feedback (stored in LangSmith)

---

## **How It Works**

1) **Planner**
   - Creates a structured outline for the essay

2) **Research (optional)**
   - Generates search queries and pulls notes using Tavily

3) **Writer**
   - Produces a draft based on plan + notes
   - Uses critique in follow-up drafts when revisions are enabled

4) **Critique**
   - Evaluates the draft and provides actionable revision guidance

5) **Research-for-critique (optional)**
   - Uses critique to request additional targeted research

6) **Revision Loop**
   - Repeats writer → critique → research until `max_revisions` is reached

---

## **Deployment**

This app is deployed on Streamlit Community Cloud:
https://essay-writer-app.streamlit.app/

To deploy your own fork:
1) Push the project to GitHub
2) Create a new Streamlit Cloud app pointing to your repo and `app.py`
3) Add secrets in Streamlit Cloud settings (same TOML keys as local secrets.toml)

---

## **Troubleshooting**

- **Missing secrets**
  - Ensure OPENAI_API_KEY is set in Streamlit secrets
  - If web research is enabled, ensure TAVILY_API_KEY is also set
  - For tracing/feedback, ensure LANGCHAIN_API_KEY is set

- **Model errors**
  - If a selected model is unavailable for your API key, switch to `gpt-4o-mini` (default) or another available model in the sidebar.

- **Large session memory**
  - Run history is stored in-session; clearing history from the sidebar can reduce memory usage during long sessions.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Contact**
- Live App: https://essay-writer-app.streamlit.app/
- For issues/feature requests: open a GitHub Issue in this repository.
- For questions or feedback, connect with me on [LinkedIn](https://www.linkedin.com/in/brejesh-balakrishnan-7855051b9/)

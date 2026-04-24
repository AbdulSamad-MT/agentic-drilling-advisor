# Agentic Drilling Advisory System — Implementation Plan

## Problem Statement

The existing `wiper-trips-predictor` project provides a real-time ML dashboard (GBT + Isolation Forest ensemble) for wiper trip risk scoring. While it produces numerical risk scores and rule-based recommendations, it lacks:

1. **Natural language reasoning** — Engineers can't ask contextual questions like *"Why did torque spike at 14:00 yesterday?"* or *"What happened last time we drilled at this depth?"*
2. **Cross-referencing capability** — The system doesn't connect current sensor patterns to historical events documented in the 163 PDF daily reports
3. **Adaptive advice** — Recommendations are hard-coded rules (`risk > 0.7 → "Perform Wiper Trip"`), not contextually synthesized from data + domain knowledge

An **agentic approach** solves all three by giving an LLM the ability to autonomously query both the sensor data and the report knowledge base, reason about the combined evidence, and produce contextual, engineer-grade advisory.

---

## Existing Assets (What We Already Have)

| Asset | Details |
|---|---|
| **Sensor CSV** | `16A(78)-32_time_data_10s_intervals.csv` — 501K rows × 36 columns, 10-second intervals |
| **Daily Reports** | `16A(78)-32_Daily_Reports/drilling/` — 163 PDF reports (Oct 2020 – Jan 2021) |
| **ML Pipeline** | `model.py` — `WiperTripPredictor` class with GBT + Isolation Forest ensemble, 77 engineered features |
| **Report Parser** | `report_parser.py` — Extracts 125 events (wiper trips, reaming, POOH, etc.) from PDFs |
| **Feature Engine** | `engine.py` — Rolling features, MSE, risk scoring, advisory generation |
| **Streamlit Dashboard** | `app.py` — Live streaming dashboard with charts and advisory panel |

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ ENGINEER (Natural Language) │
│ "Current torque is spiking and ROP dropped — should I trip?" │
└──────────────────────────────┬──────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR AGENT (LangGraph) │
│ │
│ System prompt: You are a senior drilling engineer advisor. │
│ You have access to real-time sensor data, ML risk predictions, │
│ and historical daily drilling reports for well 16A(78)-32. │
│ │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ ROUTER (Agent decides) │ │
│ │ │ │
│ │ Query type? │ │
│ │ ├── Sensor analysis → Pandas Tool │ │
│ │ ├── Historical context → RAG Report Tool │ │
│ │ ├── Risk prediction → ML Prediction Tool │ │
│ │ └── Combined analysis → Use multiple tools, synthesize │ │
│ └──────────────────────────────────────────────────────────────┘ │
└─────────┬──────────────────┬──────────────────┬────────────────────┘
 │ │ │
 ▼ ▼ ▼
┌──────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ SENSOR │ │ REPORT │ │ ML RISK │
│ ANALYSIS │ │ RETRIEVAL │ │ PREDICTION │
│ TOOL │ │ TOOL (RAG) │ │ TOOL │
│ │ │ │ │ │
│ Pandas agent │ │ ChromaDB vector │ │ WiperTrip │
│ on CSV data │ │ store of 163 │ │ Predictor │
│ (subsampled) │ │ parsed PDF │ │ .predict() │
│ │ │ report chunks │ │ ensemble score │
│ Queries: │ │ │ │ │
│ - Stats │ │ Queries: │ │ Returns: │
│ - Trends │ │ - Past events │ │ - Risk 0-1 │
│ - Anomalies │ │ - Similar conds │ │ - GBT prob │
│ - Time range │ │ - Crew actions │ │ - IF score │
│ filtering │ │ - Depth context │ │ - Top features │
└──────────────┘ └──────────────────┘ └──────────────────┘
 │ │ │
 └──────────────────┴──────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SYNTHESIZED ADVISORY │
│ │
│ "Based on the current sensor data, your torque has increased 18% │
│ over the last 10 minutes while ROP dropped 12%. The ML model │
│ shows a risk score of 0.72 (HIGH). Looking at the daily reports, │
│ a similar pattern occurred on Nov 8 at 5,200ft — the crew │
│ performed a short trip and resolved it within 2 hours. │
│ │
│ RECOMMENDATION: Perform a wiper trip. Increase flow rate by 10% │
│ while preparing. Focus on the interval 4,800-5,100ft where │
│ historical tight spots have been reported." │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| **Agent Framework** | LangChain + LangGraph | Stateful multi-tool orchestration, industry standard |
| **LLM** | OpenAI GPT-4o / GPT-4o-mini (or local via Ollama) | Strong reasoning, tool-calling support |
| **Vector Store** | ChromaDB (local, file-backed) | Lightweight, no external service needed |
| **Embeddings** | OpenAI `text-embedding-3-small` (or local `all-MiniLM-L6-v2`) | Good quality, cost-effective |
| **PDF Parsing** | PyMuPDF (already used) + Unstructured | Better table/structure extraction |
| **Data Analysis** | Pandas (already used) | Direct DataFrame queries |
| **UI** | Streamlit (already used) + `st.chat_input` | Natural chat interface |
| **Tracing** | LangSmith (optional) | Debug agent decisions |

---

## Proposed Changes

### Knowledge Base & Data Layer

#### [NEW] knowledge_base.py

PDF vectorization + ChromaDB management:

```
report_parser.py (existing)
 │
 ▼
┌─────────────────────────────┐
│ Enhanced PDF Extraction │
│ │
│ For each PDF: │
│ 1. Extract full text │
│ 2. Parse structured fields: │
│ - Report date │
│ - Depth (MD/TVD) │
│ - Operations timeline │
│ - Events & actions │
│ - Crew observations │
│ 3. Chunk into segments │
│ (~500 tokens each) │
│ 4. Add metadata: │
│ date, depth, event_type │
└──────────────┬──────────────┘
 │
 ▼
┌─────────────────────────────┐
│ ChromaDB Vector Store │
│ │
│ Collection: "daily_reports" │
│ ~500-800 chunks │
│ Metadata-filterable by: │
│ date, depth_range, │
│ event_type │
└─────────────────────────────┘
```

Functions:
- `build_report_vectorstore()` — Parse all 163 PDFs, chunk, embed, store in ChromaDB
- `query_reports(question, filters)` — Semantic search with optional metadata filters
- Persist the vector store to disk so it only needs to be built once

#### [NEW] data_tools.py

Sensor data query tools:
- `load_analysis_dataframe()` — Load and prepare CSV with named columns, datetime index
- `get_sensor_summary(time_range)` — Quick stats for a time window
- `get_anomaly_periods()` — Pre-compute interesting time periods
- `get_current_readings(idx)` — Snapshot of current sensor state

---

### Agent Tools Layer

#### [NEW] tools.py

Four tools the agent can call:

**Tool 1: `sensor_analysis`**
- Backed by a Pandas DataFrame agent (sandboxed)
- Can compute rolling averages, correlations, time-range stats
- Returns structured text summaries

**Tool 2: `report_search`**
- Backed by ChromaDB vector retrieval
- Supports metadata filtering (date, depth, event type)
- Returns relevant report excerpts with source attribution

**Tool 3: `ml_risk_prediction`**
- Wraps existing `WiperTripPredictor.predict()`
- Returns formatted risk assessment with feature importances

**Tool 4: `well_context`**
- Returns static + dynamic well information
- Helps the agent ground its responses in physical reality

---

### Agent Orchestrator

#### [NEW] agent.py

LangGraph state machine:

```
┌─────────────────────────────────────────┐
│ LangGraph State Machine │
│ │
│ START → Plan → Execute Tools → Reason │
│ │ │ │ │
│ │ ▼ │ │
│ │ [Tool Results] │ │
│ │ │ │ │
│ │ ▼ │ │
│ └── Need more info? ──┘ │
│ │ No │
│ ▼ │
│ Synthesize Advisory │
│ │ │
│ ▼ │
│ END │
└─────────────────────────────────────────┘
```

**System prompt (domain-specific):**
```
You are an AI drilling engineering advisor for well 16A(78)-32 
(Utah FORGE geothermal project). You provide expert-level 
guidance on wiper trip decisions based on real-time sensor data, 
ML risk predictions, and historical operational reports.

When advising on wiper trips, always consider:
1. Current sensor trends (torque, pressure, ROP, hookload)
2. ML model risk assessment
3. Historical precedents from daily reports
4. Well geometry and depth context
5. Operational safety margins

Structure your advice as:
- SITUATION: What the data shows
- ASSESSMENT: What it means (including historical context)
- RECOMMENDATION: Specific actions with confidence level
- MONITORING: What to watch for next
```

**Key design decisions:**
- **Temperature = 0** for deterministic tool selection
- **Max iterations = 5** to prevent infinite loops
- **Human-in-the-loop**: Agent proposes, engineer approves
- **Memory**: Conversation history maintained per session

---

### Agent Prompts

#### [NEW] prompts.py

Drilling domain-specific prompt templates:
- System prompt with drilling engineering context
- Tool selection guidance
- Output formatting templates (SITUATION → ASSESSMENT → RECOMMENDATION → MONITORING)

---

### Chat UI Integration

#### [NEW] app_agentic.py

Streamlit app with chat panel:

```
┌─────────────────────────────────────────────────────────────────┐
│ Existing Dashboard (left 70%) │ Chat Panel (right 30%) │
│ │ │
│ ┌──────────────────────────┐ │ ┌────────────────────────┐ │
│ │ Top Bar / Metrics │ │ │ Drilling Advisor │ │
│ │ Trend Charts │ │ │ │ │
│ │ Advisory Panel │ │ │ [Chat History] │ │
│ │ Event Log │ │ │ │ │
│ └──────────────────────────┘ │ │ User: Why is torque │ │
│ │ │ rising? │ │
│ │ │ │ │
│ │ │ : Based on sensor │ │
│ │ │ analysis, torque has │ │
│ │ │ increased 15% in the │ │
│ │ │ last 20 min. The ML │ │
│ │ │ model flags risk at │ │
│ │ │ 0.68. On Nov 8, a │ │
│ │ │ similar pattern at │ │
│ │ │ 5,200ft led to a │ │
│ │ │ short trip... │ │
│ │ │ │ │
│ │ │ [Quick Actions] │ │
│ │ │ Full Risk Report │ │
│ │ │ Similar Events │ │
│ │ │ Sensor Deep Dive │ │
│ │ │ │ │
│ │ │ [ Ask a question...] │ │
│ │ └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Natural language chat input (`st.chat_input`)
- Quick-action buttons for common queries
- Streaming responses (LangChain streaming callbacks)
- Agent "thinking" indicator showing which tools are being used
- Chat history persisted in `st.session_state`

---

### Pre-built Advisory Queries

| Quick Action | What the Agent Does |
|---|---|
| **"Current Risk Assessment"** | Runs ML prediction → searches reports for similar conditions → synthesizes advisory |
| **"Should I Trip Now?"** | Analyzes current trends + ML score + recent events → gives yes/no with reasoning |
| **"What Happened at This Depth?"** | Filters reports by current depth ± 500ft → summarizes past operations |
| **"Shift Handover Summary"** | Summarizes last 12 hours of sensor data + events + risk trend |
| **"Compare to Yesterday"** | Compares current parameters to same time yesterday → flags differences |

---

## Project Structure (New Files)

```
Agentic approach/
├── plan.md # This plan (also in workspace root)
├── requirements.txt # Dependencies
├── .env # API keys (OpenAI, etc.)
│
├── knowledge_base.py # [NEW] PDF vectorization + ChromaDB
├── data_tools.py # [NEW] Sensor data query tools
├── agent.py # [NEW] LangGraph orchestrator agent
├── tools.py # [NEW] Tool definitions for the agent
├── prompts.py # [NEW] System prompts + prompt templates
├── app_agentic.py # [NEW] Streamlit chat UI
│
├── chroma_db/ # [NEW] Persisted vector store
│
├── 16A(78)-32_time_data_10s_intervals.csv # Existing sensor data
└── 16A(78)-32_Daily_Reports/ # Existing PDF reports
 ├── drilling/ # 163 PDFs
 └── completion/ # Completion reports
```

---

## Dependencies (New)

```
# Core Agent
langchain>=0.3
langchain-openai>=0.3
langchain-community>=0.3
langgraph>=0.4

# Vector Store
chromadb>=0.6

# Embeddings (choose one)
sentence-transformers # For local embeddings (free)
# OR use OpenAI embeddings via langchain-openai

# Existing (carry over from wiper-trips-predictor)
streamlit
pandas
numpy
plotly
scikit-learn
pymupdf

# Optional
python-dotenv # For .env API key management
langsmith # For tracing/debugging
```

---

## User Review Required

> [!IMPORTANT]
> **LLM Provider Choice**: Which LLM do you want to use?
> - **OpenAI GPT-4o** — Best reasoning, requires API key + cost (~$0.005/query)
> - **OpenAI GPT-4o-mini** — Good enough for most queries, much cheaper (~$0.0003/query)
> - **Local via Ollama** (e.g., Llama 3, Mistral) — Free, runs offline, but slower and less capable
> - **Google Gemini** — Good alternative with generous free tier

> [!IMPORTANT]
> **Embeddings Choice**: Same question for embeddings:
> - **OpenAI `text-embedding-3-small`** — High quality, ~$0.02 per full vectorization
> - **Local `all-MiniLM-L6-v2`** via sentence-transformers — Free, offline, slightly lower quality

> [!WARNING]
> **Standalone vs. Integration**: Should this be:
> - **A) Standalone app** in the `Agentic approach/` workspace (recommended — clean separation, own requirements)
> - **B) Added to the existing `wiper-trips-predictor`** dashboard (more complex, but unified experience)

## Open Questions

> [!NOTE]
> The initial version focuses on the single well (16A(78)-32). The architecture is designed to be extensible to multi-well scenarios in the future.

> [!NOTE]
> Do you have an OpenAI API key ready, or should I design this to work with a local LLM (Ollama) first?

---

## Verification Plan

### Automated Tests
1. **Knowledge base build**: Verify all 163 PDFs are parsed → chunks created → embedded in ChromaDB
2. **Tool validation**: Test each tool independently with known queries:
 - `sensor_analysis("What is the average torque between Nov 5-10?")` → verify numeric accuracy
 - `report_search("wiper trip at 5000 feet")` → verify relevant reports returned
 - `ml_risk_prediction(data_index=1000)` → verify risk score matches existing system
3. **End-to-end**: Send 5 representative engineering questions through the full agent pipeline

### Manual Verification
- **Domain accuracy**: Review 10 advisory responses against the actual daily report ground truth
- **UI/UX**: Test the chat interface flow in the Streamlit app
- **Latency**: Ensure query-to-response time < 15 seconds for typical questions

---

## Timeline Summary

| Phase | Duration | Deliverable |
|---|---|---|
| Phase 1: Knowledge Base | 1–2 days | `knowledge_base.py`, `data_tools.py`, ChromaDB populated |
| Phase 2: Agent Tools | 1 day | `tools.py` with 4 validated tools |
| Phase 3: Orchestrator | 1 day | `agent.py` with LangGraph state machine |
| Phase 4: Chat UI | 1 day | `app_agentic.py` with integrated chat |
| Phase 5: Quick Actions | 0.5 day | Pre-built queries + polish |
| **Total** | **~5 days** | Full agentic advisory system |

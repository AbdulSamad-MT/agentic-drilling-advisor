# Agentic Drilling Advisory System — How It Works

## Overview

This project is an **AI-powered drilling advisor** that uses a **LangGraph ReAct agent** to provide **wiper trip GO/NO-GO recommendations** for well **FORGE 16A(78)-32** (Utah FORGE geothermal project, Milford, Utah).

The system combines:
- **Real-time sensor data** (36 channels, 10-second intervals)
- **Historical daily drilling reports** (163 PDFs, vectorized into a knowledge base)
- **An LLM agent** (Qwen 2.5 7B via Ollama) that reasons over both data sources

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ USER QUESTION │
│ "Should we perform a wiper trip right now?" │
└──────────────────────┬──────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────┐
│ LangGraph ReAct Agent │
│ (Qwen 2.5 7B via Ollama, local) │
│ │
│ The agent autonomously decides which tools to call │
│ and in what order, then synthesizes a final answer │
└──────┬──────────┬──────────┬──────────┬─────────────┘
 │ │ │ │
 ▼ ▼ ▼ ▼
 ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐
 │ Wiper │ │Report │ │Sensor │ │ Well │
 │ Trip │ │Search │ │Analysis│ │Context │
 │ Assess │ │(RAG) │ │ │ │ │
 └────┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
 │ │ │ │
 ▼ ▼ ▼ ▼
 ┌─────────┐ ┌────────┐ ┌──────────────────┐
 │ Sensor │ │ChromaDB│ │ Pandas DataFrame │
 │CSV Data │ │ Vector │ │ (50K+ rows) │
 │ │ │ Store │ │ │
 └─────────┘ └────────┘ └──────────────────┘
```

---

## Project Structure

```
Agentic approach/
├── 01_knowledge_base.ipynb # Step 1: Build the vector knowledge base
├── 02_agentic_advisor.ipynb # Step 2: Run the AI advisor agent
├── 16A(78)-32_Daily_Reports/ # 163 PDF daily drilling reports
├── 16A(78)-32_time_data_10s_intervals.csv # ~100MB sensor data
├── chroma_db/ # Persisted vector store (built by notebook 01)
├── requirements.txt # Python dependencies
├── .env.example # Environment variable template
├── plan.md # Project plan
└── HOW_IT_WORKS.md # This file
```

---

## What Is a Wiper Trip?

A **wiper trip** is a drilling operation where the drill string is pulled partway out of the hole and then run back to bottom. It is performed to:
- Clean the borehole of cuttings
- Condition the hole wall (prevent stuck pipe)
- Check for tight spots or swelling formations

**When is it needed?** When sensor data shows signs of:
- Rising surface torque (>10% increase)
- Rising standpipe pressure (possible pack-off)
- Dropping rate of penetration (tight hole)
- Erratic hookload (overpull/drag)
- Gas influx
- High-angle sections (>30°, where cuttings settle)

---

## Notebook 01: Knowledge Base (`01_knowledge_base.ipynb`)

> **Run this notebook ONCE** before using the advisor.

### Cell-by-Cell Breakdown

#### Cell 1 — Install Dependencies
```python
!pip install -q langchain langchain-openai langchain-community langgraph chromadb ...
```
Installs all required Python packages: LangChain (agent framework), ChromaDB (vector database), sentence-transformers (embeddings), PyMuPDF (PDF parsing), and data science libs.

#### Cell 2 — Configuration
```python
REPORT_DIR = "16A(78)-32_Daily_Reports/drilling"
CSV_PATH = "16A(78)-32_time_data_10s_intervals.csv"
CHROMA_DIR = "chroma_db"
```
Sets file paths and verifies that the PDF reports directory and sensor CSV exist.

#### Cell 3 — Define Extraction Functions
Two key functions are defined:

1. **`extract_report(filepath)`** — Opens a PDF, extracts all text, and mines metadata:
 - **Date** from "RPT DATE:" fields
 - **Depth** from "MD/TVD:" fields
 - **Events** using regex patterns (wiper trips, tight spots, stuck pipe, high torque, pack-off, drag, overpull, etc.)

2. **`chunk_text(text, chunk_size=500, overlap=100)`** — Splits report text into overlapping word-based chunks of ~375 words each, with 75-word overlap. This ensures:
 - No context is lost at chunk boundaries
 - Each chunk is small enough for effective semantic search

#### Cell 4 — Build the Vector Store
This is the main ingestion cell:

1. Creates a **ChromaDB persistent client** (stored to disk in `chroma_db/`)
2. Uses **all-MiniLM-L6-v2** sentence transformer for embeddings (384-dim vectors, runs locally, no API needed)
3. Iterates over all 163 PDF reports:
 - Extracts text and metadata
 - Splits into chunks
 - Generates unique IDs via MD5 hash (prevents duplicates)
 - Tags each chunk with detected drilling events
4. Adds all chunks to ChromaDB in batches of 100

**Output:** ~1,000+ vectorized chunks stored in `chroma_db/`

#### Cell 5–6 — Test Knowledge Base Queries
Runs sample semantic searches to verify the vector store works:
- `"wiper trip due to high torque"` — finds reports mentioning wiper trips caused by torque issues
- `"stuck pipe or pack off incident"` — finds historical stuck pipe events

#### Cell 7–8 — Load & Validate Sensor Data
Loads the 100MB CSV sensor file into a Pandas DataFrame:
- **Renames columns** from verbose names (e.g., "Top Drive Torque (ft-lbs)") to short codes (e.g., "TRQ")
- **Subsamples** every 10th row (10s → 100s intervals) to reduce memory usage
- **Cleans** sentinel values (-999.25 → NaN)
- **Forward-fills** missing values
- Shows statistical summary of 9 key drilling parameters

---

## Notebook 02: Agentic Advisor (`02_agentic_advisor.ipynb`)

> **This is the main notebook.** Run all cells top-to-bottom after Notebook 01 has built the knowledge base.

### Cell-by-Cell Breakdown

---

### Cell 1 — Setup
```python
import os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
MODEL_NAME = 'qwen2.5:7b'
```
- Imports core libraries
- Sets the LLM model to **Qwen 2.5 7B** running via Ollama (local, free, supports tool calling)
- To use a different model, change `MODEL_NAME` (e.g., `'qwen2.5:14b'`, `'llama3.1:8b'`)

---

### Cell 2 — Load Data
This cell loads **two data sources** that the agent tools will query:

**Part A — Sensor CSV:**
- Loads `16A(78)-32_time_data_10s_intervals.csv` (~50K rows after subsampling)
- Maps 20+ raw column names to short codes (WOB, ROP, RPM, TRQ, SPP, etc.)
- Cleans sentinel values and forward-fills gaps
- Time range: Nov 2020 – Jan 2021

**Part B — ChromaDB Knowledge Base:**
- Connects to the persisted `chroma_db/` directory (built by Notebook 01)
- Loads the `daily_reports` collection with all-MiniLM-L6-v2 embeddings
- This is used by the `report_search` tool for semantic search over historical reports

---

### Cell 3 — Define Agent Tools

Four tools are defined using LangChain's `@tool` decorator. The LLM agent can call any of these autonomously:

#### Tool 1: `wiper_trip_assessment()`
**The core tool.** Performs a multi-parameter risk assessment:

| Parameter | What It Checks | Risk Points |
|-----------|---------------|-------------|
| **Torque (TRQ)** | % change over last 100 data points | +25 if >10%, +10 if >5% |
| **Hookload** | % change + variability (std dev) | +20 if erratic, +8 if shifting |
| **SPP** | Standpipe pressure trend | +20 if >10%, +8 if >5% |
| **ROP** | Rate of penetration dropping | +15 if <-15%, +5 if <-5% |
| **Diff Pressure** | Spikes above 2σ threshold | +10 if spikes detected |
| **Gas** | Current vs. 2× average | +10 if elevated |
| **Inclination** | Wellbore angle | +10 if >30° |

**Decision thresholds:**
- **Score ≥ 60** → RECOMMEND WIPER TRIP
- **Score 35–59** → CONSIDER WIPER TRIP
- **Score < 35** → NO WIPER TRIP NEEDED

Returns: well info, risk score, all parameter flags, decision, and current readings.

#### Tool 2: `report_search(query)`
**RAG (Retrieval-Augmented Generation) search** over 163 historical drilling reports:
- Takes a natural language query (e.g., "wiper trips at 5000 feet depth")
- Runs semantic similarity search via ChromaDB
- Returns top 5 matching report chunks with date, depth, and event tags
- The agent uses these to find historical precedents for current conditions

#### Tool 3: `sensor_analysis(query)`
**Single-parameter deep dive:**
- Parses the query to identify which parameter to analyze (using aliases like "torque" → TRQ)
- Returns: current value, trend direction, % change, mean/std/range, anomaly count
- Used when the agent needs more detail on a specific sensor

#### Tool 4: `well_context()`
**Background information** about the well:
- Well name, location, depth, inclination
- Data coverage (rows, time range)
- Report count and sensor channel count

---

### Cell 4 — Create Agent

This cell assembles the **LangGraph ReAct agent**:

1. **LLM:** `ChatOllama(model='qwen2.5:7b', temperature=0)` — deterministic responses
2. **System prompt** instructs the agent to:
 - Always call `wiper_trip_assessment` FIRST
 - Then search historical reports for precedents
 - Correlate sensor data with historical patterns
 - Output structured sections: SENSOR STATUS → HISTORICAL PRECEDENT → RECOMMENDATION → MONITORING PLAN
3. **Memory:** `MemorySaver()` — maintains conversation history within a session
4. **`create_react_agent()`** — LangGraph's built-in ReAct (Reason + Act) loop:
 - The agent **thinks** about what tool to call
 - **Acts** by calling the tool
 - **Observes** the tool output
 - **Repeats** until it has enough information to answer

---

### Cell 5 — Ask Function
```python
from IPython.display import display, Markdown

def ask(question, thread='default'):
```
The `ask()` function:
1. Sends the question to the agent
2. The agent autonomously decides which tools to call (and in what order)
3. Collects the final response and list of tools used
4. Renders everything as **formatted Markdown** using `IPython.display.Markdown`

The `thread` parameter enables conversation memory — messages in the same thread share context.

---

### Cell 6 — Main Advisory Query
```python
response = ask(
 'Based on current sensor readings and historical report data, '
 'should we perform a wiper trip right now? '
 'Analyze sensor risk, find similar historical situations, and give a clear GO or NO-GO.'
)
```
This triggers the full advisory workflow:
1. Agent calls `wiper_trip_assessment()` → gets risk score + flags
2. Agent calls `report_search()` → finds historical wiper trip events at similar conditions
3. Agent synthesizes both into a structured GO/NO-GO recommendation

---

### Cells 7–8 — Follow-up Questions
Pre-built follow-up queries:
- Historical wiper trip events at similar depths
- Torque and hookload trend analysis

These share the same `thread='default'` so the agent remembers the previous context.

---

### Cell 9 — Interactive Mode
A `while True` loop that lets you type custom questions:
- Type any drilling question in natural language
- The agent will autonomously use its tools to answer
- Type `quit` to exit

---

## How the Agent Decides (ReAct Loop)

```
User: "Should we do a wiper trip?"
 │
 ▼
Agent THINKS: "I need to check sensor data first"
 │
 ▼
Agent ACTS: calls wiper_trip_assessment()
 │
 ▼
Agent OBSERVES: Risk score = 16/100, HOOKLOAD shifting -6.5%, SPP up 8.8%
 │
 ▼
Agent THINKS: "Risk is low but I should check historical precedents"
 │
 ▼
Agent ACTS: calls report_search("wiper trip high torque hookload")
 │
 ▼
Agent OBSERVES: Found 5 reports, dates and event descriptions
 │
 ▼
Agent THINKS: "I have enough data to make a recommendation"
 │
 ▼
Agent RESPONDS: Structured advisory with GO/NO-GO decision
```

---

## How to Run

### Prerequisites
1. **Python 3.10+** installed
2. **Ollama** installed ([ollama.com/download](https://ollama.com/download))
3. Pull the LLM model:
 ```bash
 ollama pull qwen2.5:7b
 ```
4. Install Python dependencies:
 ```bash
 pip install -r requirements.txt
 pip install langchain-ollama
 ```

### Step 1: Build Knowledge Base
1. Open `01_knowledge_base.ipynb`
2. Run all cells (takes ~2-5 minutes)
3. This creates `chroma_db/` with vectorized drilling reports

### Step 2: Run the Advisor
1. Open `02_agentic_advisor.ipynb`
2. Run all cells from top to bottom
3. The main advisory cell will output a structured wiper trip recommendation

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Qwen 2.5 7B (Ollama) | Reasoning engine, runs locally |
| Agent Framework | LangGraph (ReAct) | Autonomous tool-calling loop |
| Vector Database | ChromaDB | Semantic search over drilling reports |
| Embeddings | all-MiniLM-L6-v2 | Local sentence embeddings (384-dim) |
| PDF Parsing | PyMuPDF | Extract text from drilling report PDFs |
| Data Processing | Pandas/NumPy | Sensor data analysis |
| Tool Interface | LangChain @tool | Expose functions to the LLM agent |

---

## Sensor Parameters Reference

| Code | Full Name | Unit | Wiper Trip Relevance |
|------|-----------|------|---------------------|
| WOB | Weight on Bit | klbs | Drilling efficiency |
| ROP | Rate of Penetration | m/hr | Dropping = tight hole |
| RPM | Top Drive RPM | rpm | Drilling state indicator |
| TRQ | Surface Torque | ft-lbs | Rising = hole friction |
| SPP | Standpipe Pressure | psi | Rising = pack-off risk |
| FLOW_IN | Flow In | gpm | Circulation status |
| HOOKLOAD | Hookload | klbs | Overpull/drag indicator |
| DH_TRQ | Downhole Torque | ft-lbs | Downhole friction |
| DIFF_P | Differential Pressure | psi | Formation interaction |
| GAS | Total Gas | units | Well control indicator |
| MWD_INC | MWD Inclination | degrees | >30° = cuttings risk |

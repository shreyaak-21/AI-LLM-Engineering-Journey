# 🧠 LLM Behavior Analyzer & Interactive Chat Playground

A **research-oriented, configurable LLM experimentation platform** built with **Streamlit** and **Ollama (Mistral)** that allows users to analyze and understand Large Language Model (LLM) behavior in real time.

This project is designed **not as a toy chatbot**, but as a **foundation-level industry project** that demonstrates practical understanding of LLM internals, limitations, and controllable parameters.

---

## 🚀 Project Motivation

Modern LLM applications require more than just calling an API. Engineers must understand:

* How prompts influence behavior
* Why outputs vary with temperature
* What happens when context windows overflow
* How to handle structured outputs like JSON safely

This project was built to **experiment, analyze, and explain LLM behavior scientifically**.

---

## ✨ Key Features

### 🔹 Interactive Chat Interface

* Streamlit-based UI
* Continuous conversation loop
* Session-based memory

### 🔹 Prompt Engineering Playground

* Separate **System Prompt** and **User Prompt** inputs
* Dynamic prompt construction
* Role-based prompting

### 🔹 Generation Controls

* Temperature slider (determinism vs creativity)
* Response format selector:

  * Paragraph
  * Bullet Points
  * JSON

### 🔹 Context Window & Token Analysis

* Token counting before inference
* Context limit warnings
* Displays token usage clearly

### 🔹 Manual Chunking Strategy

* Automatically splits long inputs when context limit is exceeded
* Sequential chunk processing
* Aggregates responses safely

### 🔹 Structured Output Validation (Advanced)

* JSON extraction from messy LLM outputs
* JSON validation
* LLM-powered auto-repair for broken JSON
* Fail-safe fallback mechanism

### 🔹 Behavior Comparison Mode

* Same prompt tested at different temperatures
* Side-by-side response comparison

### 🔹 Experiment Tracking

* Logs each experiment with:

  * Prompt
  * Temperature
  * Response format
  * Token usage
  * Model used
* Stored in `experiments.jsonl`

---

## 🏗️ High-Level Architecture

```
User Input (Streamlit UI)
        ↓
Prompt Builder
(System + User + Context)
        ↓
LLM Inference Engine
(Ollama – Mistral)
        ↓
Response Post-Processing
(JSON validation / chunking)
        ↓
Experiment Logger
        ↓
UI Output
```

---


## 🧪 What This Project Demonstrates

| Concept               | How It’s Covered                   |
| --------------------- | ---------------------------------- |
| LLM inference         | Ollama-based local model calls     |
| Prompt engineering    | System vs User prompt separation   |
| Temperature control   | Adjustable creativity slider       |
| Context window limits | Token counting & overflow handling |
| Chunking              | Manual chunking & aggregation      |
| Structured outputs    | JSON enforcement & auto-repair     |
| Experimentation       | Prompt + parameter logging         |

---

## ▶️ How to Run the Project

### 1️⃣ Prerequisites

* Python 3.9+
* Ollama installed

### 2️⃣ Start Ollama

```bash
ollama serve
```

### 3️⃣ Pull Mistral Model

```bash
ollama pull mistral
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run Streamlit App

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧠 Example Use Cases

* Analyze how temperature affects determinism
* Test system prompt sensitivity
* Safely generate JSON for downstream pipelines
* Understand LLM context limitations
* Demonstrate LLM behavior in interviews or demos

---



## 🔮 Future Enhancements

* Prompt version comparison dashboard
* Schema-based JSON validation
* Model switching (OpenAI / Claude)
* Deployment on Streamlit Cloud

---

## 🏁 Conclusion

This project demonstrates **practical, production-aware understanding of LLM systems** rather than simple chatbot development. It serves as a strong foundation for roles in:

* AI Engineering
* LLM / GenAI Development
* Applied Machine Learning

---

⭐ If you found this project useful, feel free to explore and extend it!

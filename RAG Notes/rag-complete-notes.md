# RAG Learning Notes — Complete Reference

---

# 1. Embeddings — `encode_kwargs` & Normalization

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

## `encode_kwargs` ≠ text encoding
In sentence-transformers, turning text into a vector is called **"encoding"** — as in *encode this sentence into an embedding*. So `encode_kwargs` = "extra arguments to pass to the model's `.encode()` method." It has **nothing to do with UTF-8**.

## What `normalize_embeddings=True` does
It's not "turn every number under 1." It fixes the **overall length of the whole vector** — set to exactly 1.

Think of each embedding as an **arrow** pointing in some direction in space:
- The arrow's **direction** = the meaning of the text.
- The arrow's **length** = basically noise; varies with sentence length, doesn't carry meaning.

Normalizing keeps the **direction identical** but forces **every arrow to be the same length (1)**.

**Why:** cosine similarity measures the **angle** between two arrows. When every vector is length 1, that angle comparison becomes a plain dot-product — faster and numerically cleaner. Lines up perfectly with `metric="cosine"`.

Storage cost is the same either way.

## Where UTF-8 actually lives
UTF-8 is a **character encoding** — how raw bytes become text characters. It matters when *reading a file from disk*:
```python
open("data.txt", encoding="utf-8")
```
By the time text reaches the embedding model, it's already a Python `str` (Unicode internally), so there's no byte-decoding step left.

## The pipeline
```
"Robbers broke into the bank"   ← text (UTF-8 relevant ONLY if read from a file)
            │
            ▼   model.encode()   ← the actual embedding step
   [0.42, -1.9, 0.03, ...]       ← raw embedding (384 floats)
            │
            ▼   normalize_embeddings=True   ← optional rescale to length 1
   [0.21, -0.95, 0.015, ...]     ← final vector stored in the vector DB
```

| Term | Meaning | Where it applies |
|------|---------|------------------|
| **Character encoding** (UTF-8) | bytes ⇄ text characters | reading/writing files |
| **`encode_kwargs`** | text → embedding vector | the embedding model |

---

# 2. Chunking Methods

## The 3 fixed-size splitters

| Splitter | How it cuts | Typical settings |
|----------|-------------|------------------|
| **CharacterTextSplitter** | Blindly every N characters | `chunk_size=200, overlap=20` |
| **RecursiveCharacterTextSplitter** | Tries natural boundaries first (`\n\n` → `\n` → space → char), still respecting a size cap | `chunk_size=500, overlap=50` |
| **TokenTextSplitter** | Every N tokens (LLM tokens, not characters) | — |

**Note:** `chunk_size` counts **characters** (or **tokens**), not words. `chunk_size=500` ≈ ~80–100 words.

## The one that's not purely blind
`RecursiveCharacterTextSplitter` doesn't just cut at position 500 — it **tries to cut at a paragraph break, then a line break, then a space**, chopping mid-word only as a last resort. It respects *structure* (whitespace/punctuation) but knows **nothing about meaning**.

## The progression
```
CharacterTextSplitter          → cut by raw character count (dumbest)
RecursiveCharacterTextSplitter → cut by count, but prefer natural separators
TokenTextSplitter              → cut by token count (LLM-aware sizing)
──────────────────────────────────────────────────────────────────────────
SemanticChunker                → cut by MEANING (embeddings + cosine threshold)
```

---

# 3. Semantic Chunking

## Definition
> Semantic chunking = splitting a document into meaningful units (chunks) based on **semantic similarity** — not just by number of tokens or lines.

## Why it matters
> Better chunks → better retrieval → better grounding → better answers.

Chunks should be **self-contained, contextually rich, and logically separated.**

## The problem with fixed-size
Fixed-size chunking is blind to content. It'll cut a chunk off mid-thought:
```
Chunk 1: "...The engine overheated because the coolant"
Chunk 2: "level was low. To fix this, first turn off..."
```
One idea sliced across two chunks → neither retrieves cleanly.

## How it works — the 5 steps

| Step | What happens |
|------|--------------|
| **1. Document Segmentation** | Document → sentences or paragraphs |
| **2. Sentence Embedding** | Each sentence → vector (OpenAI, HF ⇒ Sentence Transformers) |
| **3. Semantic Similarity Check** | Cosine similarity between *adjacent* embeddings |
| **4. Merging of Sentences** | Merge adjacent sentences if semantically similar |
| **5. Form Chunks** | Merge S1+S2 → chunk 1; S3 → chunk 2 |

## The threshold
```
S1 ↔ S2  ⇒ similar ⇒ 0.95
S2 ↔ S3  ⇒ 0.75
S3 ↔ S4  ⇒ 0.95
   Threshold ⇒ 0.8
```
Any adjacent pair scoring **above** the threshold stays together; a pair **below** it (0.75) is a **cut point** — a new chunk starts.

## Worked example
**Input — 4 sentences:**
```
1. LangChain is a framework for building LLM-powered apps.
2. It integrates with tools like OpenAI and Pinecone.
3. The Eiffel Tower is located in Paris.
4. France is a popular tourist destination.
```
**Output after semantic chunking:**
```
[ "LangChain is a framework...", "It integrates with tools..." ]   ← 1+2 (both about LangChain)
[ "The Eiffel Tower is located in Paris." ]                        ← 3 (topic shift)
[ "France is a popular tourist destination." ]                     ← 4 (different again)
```

## Fixed-size vs Semantic

| | Fixed-size | Semantic |
|---|---|---|
| **Splits on** | character/token count | meaning (embedding similarity) |
| **Chunk size** | uniform | variable — as long as the idea runs |
| **Cuts mid-idea?** | often | rarely |
| **Cost/speed** | instant, free | slower — must embed every sentence |
| **Setup** | trivial | needs an embedding model + a threshold |

## The code — how it works

1. **Split** — break on newlines, drop empty lines. *(Here a "sentence" = a line.)*
2. **Embed** — each sentence → vector via the HF model.
3. **Threshold = 0.7** — the knob for how tight chunks are.
4. **The loop** — compare each sentence to the one right before it (`i-1` vs `i`, adjacent pair).
   - If `sim ≥ 0.7` → append to the **current** chunk (keep it going).
   - If `sim < 0.7` → **close** the current chunk (push it into `chunks`) and **start a new one** with this sentence.
5. **Final append** — after the loop, the last open chunk still needs to be pushed into `chunks`, or you'd lose it.

**In one line:** split by newline → embed each line → walk through adjacent pairs, keep merging while similarity ≥ 0.7, and cut a new chunk whenever it drops below.

> **Threshold intuition:** higher = stricter = more, smaller chunks; lower = looser = fewer, bigger chunks.

## The class wrapper

| Method | Input → Output | Use |
|--------|----------------|-----|
| `split` | `str` → `list[str]` | quick, raw text |
| `split_documents` | `list[Document]` → `list[Document]` | fits the real RAG pipeline (loaders → splitter → vector store all speak `Document`) |

> `split` is the brain, `split_documents` is the adapter that lets your custom chunker drop into the same slot where you'd normally put `RecursiveCharacterTextSplitter`.

---

# 4. Hybrid Search — Dense & Sparse Retrieval

## ① Sparse Retrieval = keyword matching
- Matches **exact words** using old-school NLP: **TF-IDF, BM25, BoW** (bag of words).
- Turns text into a **sparse matrix** — mostly zeros, a `1` where a word exists.
- Example: `"I want to have food"` → `[1, 0, 0, 0, 1]` (1s mark which known words appear).
- Great at **exact match**, dumb about meaning. Searching "car" won't find "automobile."

## ② Dense Retrieval = meaning matching
- Uses **vector embeddings + cosine similarity**.
- Stored in **FAISS / Chroma**.
- Finds **semantically similar** sentences even if the words differ. "car" *does* find "automobile."
- Great at meaning, but can miss when you need a **precise keyword** (like a product code or a name).

## Sparse vs Dense

| Aspect | Sparse Retrieval | Dense Retrieval |
|--------|------------------|-----------------|
| Matches on | Exact words / keywords | Meaning (semantics) |
| Technique | TF-IDF, BM25, BoW | Embeddings + cosine similarity |
| Representation | Sparse matrix (mostly 0s) | Dense vector (all floats) |
| Tools | rank_bm25, scikit-learn | FAISS, ChromaDB |
| Good at | Precise keyword lookup | Synonyms, paraphrases |
| Blind spot | No semantic understanding | Can miss exact keywords |

## Why combine them → Hybrid
Each one has a blind spot, so you just... use both.
> Sparse gives you **exact keyword match**, dense gives you **semantic power** — hybrid = best of both worlds.

## The formula
```
Score_hybrid = α × Score_dense + (1 − α) × Score_sparse
```
- **Score_dense** = cosine similarity (input vs vector store)
- **Score_sparse** = TF-IDF score (input vs keyword representation)
- **α (alpha)** = the weight knob, here `α = 0.5`

α is "how much do I trust meaning vs keywords." `0.5` = trust both equally. Crank α up → lean semantic; crank it down → lean keyword.

## Worked example
Query: **"build application using LLM"**

| Doc | Dense (cosine) | Sparse (TF-IDF) | Hybrid = 0.5×dense + 0.5×sparse |
|-----|----------------|-----------------|---------------------------------|
| D1 "LangChain helps build LLM apps" | 0.85 | 0.60 | **≈ 0.82** ✅ |
| D2 "Pinecone for vector search"     | 0.40 | 0.20 | ≈ 0.18 |
| D3 "Eiffel Tower is in Paris"       | 0.10 | 0.10 | ≈ 0.04 |

**D1 wins big** because it scores high on **both** — semantically about building LLM apps AND literally contains the keywords "build," "LLM," "apps."

## In LangChain

| Notes concept | LangChain class |
|---------------|-----------------|
| Sparse retrieval (TF-IDF/BM25) | `BM25Retriever` |
| Dense retrieval (embeddings+cosine) | `vectorstore.as_retriever()` |
| Hybrid fusion (α weighting) | `EnsembleRetriever(weights=[...])` |

```python
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]   # ← this is your α
)
```

**Nuance:** LangChain's `EnsembleRetriever` uses **RRF (Reciprocal Rank Fusion)** under the hood — it blends by **rank position**, not raw score. BM25 scores (0–20+) and cosine scores (0–1) live on different scales, so you can't add them directly. Same spirit as the formula, cleaner math.

---

# 5. Re-Ranking

## The core idea
Retrieval is **two-stage**:
1. **Fast retriever** (BM25 / FAISS / hybrid) grabs top-k docs quickly — cheap but rough.
2. **Slower, smarter model** (cross-encoder or LLM) **re-scores and reorders** those k docs by relevance to the query.

> Get candidates fast → then carefully re-sort them so the best one lands on top.

## Why not just use the accurate model directly?
Because it's expensive. You can't run a cross-encoder over 100k docs. So: **cheap retriever narrows 100k → 10, then the expensive model perfectly orders those 10.**

## The flow
```
                          Input Query
                               │
                               ▼
                        ┌─────────────┐
                        │ Vectorstore │
                        └─────────────┘
                         ╱           ╲
                   Exact              Similar
                      ╱                   ╲
              ┌──────────┐        ┌──────────────────┐
              │  BM25    │        │ Semantic Search  │ → FAISS
              └──────────┘        └──────────────────┘
                    │                       │
                  Top K                   Top K
                    │                       │
                    └────────┬──────────────┘
                             ▼
                     ┌───────────────┐
                     │ HYBRID SCORE  │
                     └───────────────┘
                             │
                    Top K / Relevant chunks
                             ▼
        ┌────────────────────────────────────────┐
        │            RE-RANKER                   │   ← LLM re-ranks the order
        │  (cross-encoder / LLM re-scores)       │      based on the query
        └────────────────────────────────────────┘
                             │
                    Reordering => Rank
                     1,2,3,4,5 → 5,2,1,4,3
                             ▼
                        ┌─────────┐
                        │   LLM   │
                        └─────────┘
                             │
                             ▼
                    Output => Accurate answer
```

## Stage by stage

| Stage | Component | What it does |
|-------|-----------|--------------|
| **Input** | Query | User question enters the pipeline |
| **Path A — Exact** | **BM25** | Keyword matching (TF-IDF / BoW) on a **sparse** representation → Top K |
| **Path B — Similar** | **Embeddings + FAISS** | Semantic search via cosine similarity on **dense** vectors → Top K |
| **Fusion** | **Hybrid Score** | Blends both score lists by weight (α) → combined Top K |
| **Refinement** | **Re-Ranker** | Cross-encoder / LLM re-scores each **query-doc pair** and reorders |
| **Generation** | **LLM** | Generates the final answer from the reranked context |

## Key points
- **Two parallel paths:** the query goes to BM25 *and* to the embedding/FAISS side independently — BM25 has its own keyword index, it does not read the vector store.
- **Sparse vs Dense** are how docs are *represented*. Each retriever *outputs* a ranked Top-K with scores. Those scores get blended.
- **The reorder is the whole point:** retriever order `1,2,3,4,5` becomes `5,2,1,4,3` — the doc originally ranked 5th was actually the most relevant.
- **Re-ranker ≠ final LLM:** the re-ranker is a separate scoring stage; the final LLM only generates the answer.

## The 3 stages
```
1) Retrieval  →  2) Re-Ranking  →  3) Generation
```

## Bi-encoder vs Cross-encoder
- **Bi-encoder (retrieval):** embeds query and doc *separately* → compare vectors. Fast, but shallow.
- **Cross-encoder (reranking):** feeds query **and** doc together into the model → outputs a relevance score. Way more accurate, way slower — which is why it only runs on the top-k.

## Why it matters
- **Relevance** — top-k from a retriever is often only *loosely* related; reranker fixes the order.
- **Less hallucination** — irrelevant docs get filtered → grounded answers.
- **Query intent** — cross-encoder reads the **query+doc together**, so it understands intent.
- **Noise reduction** — junk chunks get pushed to the bottom.

> Retriever = fast but sloppy. Reranker = slow but precise. Use the retriever to shortlist, the reranker to sort.

---

# 6. MMR — Maximal Marginal Relevance

## The problem it solves
Your retriever returns top-k by similarity... and the top 3 all say basically the **same thing**. You wasted your context window on duplicates.

```
1. Langchain GenAI
2. Langchain Agentic AI
3. Langchain GenAI and applications   ← redundant with #1
```

## What MMR does
Picks docs that are **both**:
1. **Relevant** to the query ✓
2. **Diverse** from each other (non-redundant) ✓

> It stops the retriever from returning near-identical docs that repeat the same content.

## The formula
```
MMR(d) = λ × Sim(d, q) − (1−λ) × max Sim(d, s)
                                   s∈S
```
- `Sim(d, q)` = how relevant the doc is to the **query** → *reward*
- `max Sim(d, s)` = how similar it is to **already-selected** docs → *penalty*
- `λ` (0 to 1) = the dial: **high λ → relevance**, **low λ → diversity**

So: **score = relevance minus redundancy.**

## Worked example (λ = 0.7)

**Step 1** — pick the most relevant doc outright:

| Doc | Sim to query |
|-----|--------------|
| **D1** | 0.95 ✅ picked |
| D2  | 0.93 |
| D3  | 0.80 |

**Step 2** — score D2 and D3 *with the redundancy penalty*:
```
Sim(D1,D2) = 0.90  ← very redundant
Sim(D1,D3) = 0.30  ← diverse
```
```
MMR(D2) = 0.7×0.93 − 0.3×0.90 = 0.651 − 0.27 = 0.381
MMR(D3) = 0.7×0.80 − 0.3×0.30 = 0.560 − 0.09 = 0.470  ✅ wins
```

**D3 wins despite being *less* relevant** (0.80 < 0.93), because D2 was nearly a duplicate of D1.

| Rank | Document | Reason |
|------|----------|--------|
| 1 | D1 | Highest relevance |
| 2 | D3 | Best diversity + relevance (MMR) |

## When to use MMR
- **RAG** — avoid feeding the LLM redundant documents
- **Chatbots** — FAQ, search apps, document browsers
- When your retriever **already returns many results** and you want coverage
- Pairs great with **hybrid retrieval** (dense + sparse)

## When NOT to use MMR

| Scenario | Why you might skip MMR |
|----------|------------------------|
| Extremely short context window | You may just want top-1 most relevant |
| You need **precision only** | Not focused on coverage |
| Documents are already diverse | No need to enforce diversity |
| You're already **reranking with an LLM** | Redundancy handled by post-filtering |

## In LangChain
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.7}   # ← your λ
)
```

> **Reranker** = "put the most relevant on top."
> **MMR** = "don't give me five copies of the same answer."

---

# 7. Query Enhancement — Query Expansion Technique

## The core insight
Everything before this (hybrid search, reranking, MMR) improves the **retrieval side**.
Query expansion is the first technique that fixes the **input side** — the query itself.

> In a RAG pipeline, the quality of the query sent to the retriever determines how good the retrieved context is — and therefore how accurate the LLM's final answer will be.

## What is Query Enhancement?
Techniques used to **improve or reformulate the user query** to retrieve better, more relevant documents from the knowledge base (vector store).

**Especially useful when:**
1. The original query is **short, ambiguous, or under-specified**
2. You want to **broaden the scope** to catch synonyms, related phrases, or spelling variants

## The problem
```
"LangChain memory"  →  too vague, misses relevant docs
"tools in LLM"      →  under-specified
"retrieval"         →  one word, way too broad
```

## The fix
Run the query through an **LLM + prompt** to rewrite/expand it before searching:

| Problem (Original Query) | Solution (Enhanced Query) |
|--------------------------|---------------------------|
| "LangChain memory" | "LangChain memory modules, conversation memory" |
| "tools in LLM" | "LangChain tools, APIs, calculator, agent tools" |
| "retrieval" | "vector retrieval, dense search, BM25, MMR" |

Now the retriever has synonyms, related phrases, and spelling variants to match against → catches docs it would've missed.

## How Query Expansion Works
```
                                    ┌─────────────┐
                                    │ VectorStore │
                                    └─────────────┘
                                       ↑        │
Query ──────┐                  Enhanced│        ↓
            ↓                    Query ┌──────────────────┐
   ┌──────────────────┐  ────────────→ │    Retriever     │
   │ Query Enhancement│                │  FAISS / HYBRID  │
   └──────────────────┘                └──────────────────┘
            ↑                                   │
   Enhancing the query                  Top K Documents
            │                                   ↓
   ┌────────────────────┐              ┌─────────────┐
   │ LLM + Prompt=chain │              │  Re-Ranker  │
   └────────────────────┘              └─────────────┘
                                               │
                                    Re-Rank Top K Documents
                                        (Context)
                                               ↓
                                          ┌─────────┐
                                          │   LLM   │
                                          └─────────┘
                                               ↓
                                             Output
```

**Key point:** query enhancement is a **pre-processing step** that slots in *before* everything else in the pipeline.

## The full stack now
```
expand query → hybrid retrieve → rerank → (MMR) → generate
```

> **Better queries → better retrieved chunks → better grounded LLM answers.**
>
> Hybrid/rerank/MMR fix *how you search*. Query expansion fixes *what you search for*.

---

# 8. Query Enhancement — Query Decomposition

## The core idea
Query expansion (#7) enriches **one** query. Query decomposition goes the other way: it **splits one complex, multi-part question into several simpler, atomic sub-questions** — each retrieved and answered on its own, then stitched back together.

> Query decomposition = take a **complex, multi-part** question → break it into simpler **atomic sub-questions** that can each be retrieved and answered individually.

## Why use it
- **Complex queries pack multiple concepts** into one sentence.
- **Retrievers/LLMs miss parts** — a single embedding of a two-part question leans toward the dominant part and drops the rest.
- **Enables multi-hop reasoning** — answer in steps.
- **Allows parallelism** — sub-questions fan out independently (great in multi-agent frameworks).

## Two ways to split the query

| # | Method | How |
|---|--------|-----|
| 1 | **LLM + prompt** | Ask an LLM to rewrite the complex query into a list of sub-questions |
| 2 | **Regex operation** | Cheap split on punctuation (`. , ! ?`) — no LLM call needed |

## Worked example
**Complex query:**
> "What memory modules does LangChain support, and how are they different from CrewAI?"

**Decomposer → 3 sub-questions:**
```
Sub Q1: "What memory modules does LangChain support?"
Sub Q2: "CrewAI agents / memory"
Sub Q3: "LangChain memory vs CrewAI agents"
```

## How Query Decomposition works
```
                          Complex Query (q)
                                │
                       ┌────────┴────────┐
                       │   Decomposer    │   ← LLM+prompt  OR  regex [.,!?]
                       └────────┬────────┘
          ┌─────────────────────┼─────────────────────┐
        Sub Q1                Sub Q2                 Sub Q3
          │                     │                      │
       Retriever             Retriever              Retriever
          │  Top K              │  Top K               │  Top K
       Context               Context                Context
          │                     │                      │
       LLM Call              LLM Call               LLM Call     (+ prompt)
          │  O1                 │  O2                  │  O3
          └─────────────────────┼──────────────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │ Answer Combiner / Synthesizer │
                 └──────────────────────────────┘
                                │
                                ▼
                           Final Answer
```
Each sub-question runs the **full retrieve → LLM** loop on its own; a final **synthesizer** merges `O1 + O2 + O3` into one grounded answer.

## Expansion vs Decomposition

| | Query Expansion (#7) | Query Decomposition (#8) |
|---|---|---|
| **Does what** | Enriches **one** query with synonyms/related terms | Splits **one** query into **many** sub-queries |
| **Output** | 1 wider query | N atomic sub-questions |
| **Best for** | Short / vague / under-specified queries | Complex, multi-part questions |
| **Retrieval** | one pass | one pass **per** sub-question |

## Major disadvantage
**Many LLM calls and retrieval calls.** One question becomes N sub-questions, each costing a retrieval + an LLM call, plus a final synthesis call — so latency and cost scale with the number of sub-questions.

> Expansion widens the net; decomposition breaks a hard question into easy ones — at the price of more calls.

---

# 9. Multimodal RAG

## The core idea
Normal RAG only retrieves and reasons over **text**. Multimodal RAG extends the pipeline to **text *and* images** — so a query can pull the relevant paragraph *and* the relevant chart/photo from your documents, and the LLM answers using both.

> Same RAG skeleton (chunk → embed → store → retrieve → generate) — but every stage now handles images alongside text.

## The two pieces that make it "multimodal"

| Piece | Job |
|-------|-----|
| **CLIP** (embeddings) | *Contrastive Language–Image Pre-training* — embeds text **and** images into **one shared vector space**, so a text query can match image content. Built from a **Vision Transformer (images) + Transformer (text)**. |
| **Multimodal LLM** (generation) | An LLM that reads text **and** images to write the answer — e.g. OpenAI `gpt-4.1`, Google `gemini-flash-2.5`. |

## The pipeline
```
INGEST                                   QUERY
PDF / doc                                Query
   │ extract                               │ CLIP embed
   ├─ text  → chunk → embed ┐              ▼
   └─ images ──────→ embed ─┤          Retriever ──► top-K (text + image)
                            │              │
                     (CLIP embeddings)     ▼  format
                            ▼          Multimodal LLM  (gpt-4.1 / gemini)
                          FAISS  ◄─────────┘          │
                     (vector store)                   ▼
                                               Multimodal Answer
```
1. **Extract** text *and* images from each document.
2. **Embed both** with **CLIP** (images often passed as base64) → one shared space.
3. **Store** the vectors in **FAISS** (or any vector store).
4. **Query** → CLIP-embed → retrieve the top-K **text + image** matches.
5. **Generate** — feed the retrieved text and images to a **multimodal LLM** → a grounded answer that uses both.

## Why it matters
- Answers questions whose evidence lives in **figures, tables, screenshots, or scanned pages** — not just prose.
- **CLIP's shared space** is the key trick: text and images become comparable, so one query hits both modalities at once.

> Text-only RAG reads the words; multimodal RAG also *sees the pictures*.

---

# Appendix A — LangChain 1.x Import Migration

LangChain 1.x split the monolith into packages. Course material using `from langchain.<something>` is almost certainly outdated.

| Old (broken) | New (LangChain 1.x) |
|--------------|---------------------|
| `langchain.schema` → `Document` | `langchain_core.documents` |
| `langchain.schema.runnable` | `langchain_core.runnables` |
| `langchain.prompts` | `langchain_core.prompts` |
| `langchain.text_splitter` | `langchain_text_splitters` |
| `langchain.document_loaders` | `langchain_community.document_loaders` |
| `langchain.vectorstores` | `langchain_community.vectorstores` |
| `langchain.retrievers` → `EnsembleRetriever` | `langchain_classic.retrievers` |
| `langchain.chains.combine_documents` | `langchain_classic.chains.combine_documents` |
| `langchain.chains.retrieval` | `langchain_classic.chains.retrieval` |
| `langchain_openai` → `OpenAIEmbeddings` | `langchain_huggingface` → `HuggingFaceEmbeddings` |

**The rule of thumb:**
- **`langchain_core`** → base types: `Document`, `PromptTemplate`, output parsers, runnables
- **`langchain_community`** → integrations: loaders, vectorstores, `BM25Retriever`
- **`langchain_text_splitters`** → all splitters
- **`langchain_classic`** → legacy chains + `EnsembleRetriever`
- **`langchain`** → only a few things left, like `init_chat_model`

**Still correct:**
```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
```

---

# Appendix B — Setup Reference

## Environment variables (`.env`)
```
GROQ_API_KEY=...
ASTRA_DB_APPLICATION_TOKEN=...
ASTRA_DB_API_ENDPOINT=...
PINECONE_API_KEY=...
```

Load them with:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Don't** write `os.environ["X"] = os.getenv("X")` — it's a no-op that reads a value and writes it back. If the var doesn't exist it returns `None` and crashes with `TypeError: str expected, not NoneType`.

**Watch the spelling:** `GROQ` (the fast inference company) ≠ `GROK` (xAI's model).

## The LLM
```python
llm = init_chat_model("groq:llama-3.3-70b-versatile", temperature=0.4)
```
`gemma2-9b-it` has been decommissioned on Groq.

**Current Groq options:**
| Model | Good for |
|-------|----------|
| `groq:llama-3.3-70b-versatile` | **Recommended** — strong, general-purpose, great for RAG |
| `groq:llama-3.1-8b-instant` | Faster/cheaper, lighter reasoning |
| `groq:openai/gpt-oss-20b` | Non-Llama alternative |

## `temperature`
Controls how **random vs. deterministic** the output is.

| Temperature | Behavior |
|-------------|----------|
| **0.0** | Fully deterministic — always picks the most likely word. Same input → same output. |
| **0.4** | Mostly focused, slight flexibility. Reliable but not robotic. |
| **0.7–1.0** | More creative/varied. Good for brainstorming, writing. |
| **>1.0** | Wild, sometimes incoherent. |

For RAG you want the model **faithful to the documents**, so keep it low (0.0–0.4) to reduce hallucination and keep answers consistent.

> **temperature = how much the model is allowed to improvise.**

## Embedding dimensions
`all-MiniLM-L6-v2` produces **384-dim** vectors. If you create a Pinecone index, `dimension` must be **384**, not 1024 (that's `text-embedding-3-small`'s config). A mismatch means inserts fail.

## Required packages
```bash
pip install -qU langchain-community langchain-text-splitters langchain-huggingface
pip install -qU faiss-cpu rank_bm25 sentence-transformers python-dotenv
```

## Vector DB free tiers

| | Free tier good enough for learning? | Self-host to avoid cost? |
|---|---|---|
| **Pinecone** | Yes (Starter plan) | No (cloud-only, proprietary) |
| **AstraDB** | Yes (generous free tier) | No (but built on open-source Cassandra) |
| **Chroma / FAISS / Qdrant** | Fully free | Yes — run locally, $0 forever |

**Pinecone vs AstraDB:** Pinecone is a purpose-built, vector-native, proprietary database — vectors are the primary object. AstraDB is Apache Cassandra (DataStax) with vector search added, so you get full CQL and can keep app data + embeddings in one store. Pick Pinecone for a focused best-in-class vector index; pick AstraDB for open-source foundations and one DB for everything.

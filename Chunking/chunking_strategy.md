# Metadata-Augmented Hybrid Chunking Strategy

This document provides a detailed technical explanation of the chunking strategy implemented for the Legal RAG Document project.

## 1. Technique Overview
**Name:** Metadata-Augmented Hybrid Chunking with Parent-Child Inheritance  
**Goal:** To break down large legal texts into retrievable units while preserving critical context (Source Filename and Paragraph IDs) for every single chunk, effectively "de-coupling" the context from the physical text location.

## 2. How It Works (The Logic)

The strategy operates in a two-stage process: **Macro-Segmentation** (identifying logical blocks) followed by **Micro-Segmentation** (managing size constraints).

### Step 1: Strategy Selection (Hybrid Approach)
The script first analyzes the file content to decide the best splitting strategy:
*   **Strategy A (Bullet-Based):** If the file contains the specific bullet character `■`, the system assumes this denotes a logical section start. It splits the document primarily by this bullet.
*   **Strategy B (Tag-Based):** If no bullets are found (e.g., `10. Azadi...`), the system splits the document using high-level regex targeting `[Para ...]` tags as the natural boundaries.

### Step 2: Macro-Segmentation & Metadata Extraction
Once a logical block is isolated (e.g., everything belonging to "Para 26"), the system extracts the metadata *before* processing the text further.
*   **Extraction Regex:** `r'\[Para\w*.*?\].*?$'` or similar patterns are used to find tags like `[Para 20]`, `[Paras 133 to 135]`.
*   **Cleaning:** The extracted tag is removed from the raw text to prevent duplication in the embedding, but it is saved aside as the `para_id` for that block.

### Step 3: Micro-Segmentation (Sub-Chunking)
Often, a single logical paragraph (e.g., Para 133) is too large for a single LLM context window or embedding vector. We break it down further *without* losing the label.
*   **Sentence Splitting:** The text is split into sentences using punctuation markers (`.`, `?`, `!`).
*   **Accumulation:** Sentences are grouped together sequentially until they reach the `max_chars` limit.
*   **Size limit:** 1200 characters.

### Step 4: Metadata Inheritance
This is the most critical feature.
*   If "Para 133" is split into 3 smaller chunks because it was 3000 characters long, **all 3 chunks** receive the metadata `para_id: "[Para 133]"`.
*   This ensures that no matter how granular we go, the RAG system always knows exactly which legal paragraph a specific sentence came from.

## 3. Technical Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Chunking Utility** | Custom Python Script | `chunking.py` |
| **Max Chunk Size** | `1200` characters | The hard limit for creating a new sub-chunk. |
| **Overlap** | `0` (None) | Current implementation flushes the buffer completely when the limit is reached; there is no sliding window overlap. |
| **Split Separators** | `■`, `[Para ...]` | Primary delimiters for logical sections. |
| **Sub-splitter** | Sentence Boundary | Splits by `.`, `?`, `!` to ensure chunks don't break in the middle of a sentence. |

## 4. Why This Approach?

Standard fixed-size chunking (e.g., "every 500 characters") often fails in legal documents because:
1.  It cuts off in the middle of sentences.
2.  **Crucially:** It loses the "Label". If you split Para 20 into two parts, the second part is just floating text with no ID.

**Our approach ensures:**
*   **Precision:** Every retrieved chunk allows the LLM to cite its source precisely ("As seen in Para 20...").
*   **Coherence:** Chunks obey sentence boundaries, making them semantically complete.
*   **Flexibility:** Handles different file formats (with or without bullets) automatically.

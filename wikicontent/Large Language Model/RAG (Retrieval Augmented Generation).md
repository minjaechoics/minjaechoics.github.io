⁉️ **How can we make an LLM answer content it was not trained on?**  
⁉️ **Can’t we just keep teaching it new knowledge repeatedly?** ⇒ This causes *catastrophic forgetting* of existing knowledge.

## Retrieval System

#### Web Search
**PageRank**: Importance = the average importance of other websites that include a link to a given website.

$$
PR(A) = \frac{1 - d}{N} + d\left(\frac{PR(B)}{L(B)} + \frac{PR(C)}{L(C)} + \frac{PR(D)}{L(D)} + \cdots\right)
$$

- $d$: damping factor  
  $(1-d)$ represents the probability of jumping from a random page rather than a linked page  
- $PR(A)$: the probability that a user ends up on page A

⁉️ **Can we guarantee that the retrieved documents match the context of the question?** No!

#### Text-based Retrieval
**BM25** (Raw text-based retrieval)  
Example: the more overlapping words, the more likely the document is retrieved.

- $Q$: query (containing keywords $q_i$)  
- $N$: total number of documents  
- $n(q_i)$: number of documents containing $q_i$

$$
\mathrm{score}(D,Q)=\sum_{i=1}^{n}\mathrm{IDF}(q_i)\cdot
\frac{f(q_i,D)\cdot (k_1+1)}{f(q_i,D)+k_1\cdot\left(1-b+b\cdot\frac{|D|}{\mathrm{avgdl}}\right)}
$$

$$
\mathrm{IDF}(q_i)=\ln\left(\frac{N-n(q_i)+0.5}{n(q_i)+0.5}+1\right)
$$

⁉️ **Isn’t it hard to capture context using only individual words?**

#### Dense Retrieval
**Dense Passage Retrieval (DPR)** (Dense retrieval)  
Uses two sentence encoders (e.g., BERT models).

$$
sim(p,q)=E_Q(q)^T E_P(p)
$$

Similarity between query $p$ and passage $q$ = inner product of two vectors!

$$
\mathcal{D} = \{(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-)\}_{i=1}^{m}
$$

- $p^+$: labeled positive data  
- $p^-$: labeled negative data (hard negatives + in-batch negatives)

**Contrastive loss**:
$$
L(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-) = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}}
$$

⁉️ **But creating labeled data is extremely expensive!**

**Contriever**: A pipeline that trains dense retrieval models *without human supervision*.  
**Idea**: *Sections from the top and bottom of the same page (e.g., Wikipedia) are positively related.*  
One part is treated as the query, another as the positive document, and content from other pages is treated as negative.

- $k_+$: positive document (from the same document)  
- $k_i$: negative documents (from different documents)

$$
\mathcal{L}(q,k^+) = - \frac{\exp(s(q,k^+)/\tau)}{\exp(s(q,k^+)/\tau) + \sum_{i=1}^{K} \exp(s(q,k_i)/\tau)}
\qquad
s(q,d) = \langle f_\theta(q), f_\theta(d) \rangle
$$

---

## Improving RAG at the Inference Level

#### What if too many documents are retrieved?
**REPLUG (REtrieval and PLUG)**  
**Assumption**: *The LLM is a black box!* (model parameters are unknown)

⁉️ **LLMs have a maximum context length. What if the documents exceed it?**  
Instead of showing all documents at once, **split them**, generate **multiple responses**, and then **aggregate them at the response level**.  
(Since documents have different importance, responses should be weighted accordingly.)

$$
p(y \mid x, \mathcal{D}') = \sum_{d \in \mathcal{D}'} p(y \mid d \circ x) \cdot \lambda(d, x)
$$

- **$\lambda(x,d)$**: similarity score from the retrieval model  
$$
\lambda(d, x) = \frac{e^{s(d, x)}}{\sum_{d \in \mathcal{D}'} e^{s(d, x)}}
$$

⁉️ **LLMs and retrievers are not jointly trained, so performance is limited.**  
Example: among four documents, the *second* may be the most important in reality, but after REPLUG the *third* might be judged most important.

**LSR Algorithm**: Train a small retrieval system from the LLM’s perspective!

$$
P_R(d \mid x) = \frac{e^{s(d, x)/\gamma}}{\sum_{d \in \mathcal{D}'} e^{s(d, x)/\gamma}}
$$

$$
Q(d \mid x, y) = \frac{e^{P_{LM}(y \mid d, x)/\beta}}{\sum_{d \in \mathcal{D}'} e^{P_{LM}(y \mid d, x)/\beta}}
$$

$$
\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} KL(P_R(d \mid x) \parallel Q_{\text{LM}}(d \mid x, y))
$$

---

#### Retrieval quality depends on the query! Creating better queries
When the actual query differs from what the retriever saw during training, performance can degrade.

🧐 When a question is given to an LLM, it can **generate documents related to the question without external resources**!

**HyDE**: After asking a question, GPT generates a hypothetical answer, which is then used as input to the retrieval system.

$$
sim(q, p) = \left( \sum_{k=1, \dots, K} E_P(\tilde{p}_k(q))/K \right)^T E_P(p)
$$

- Uses **Contriever**
- *“Transform the input query!”*

**LAME-R** (Language Model Augmented Embeddings for Retrieval)
- Uses **BM25**
- First retrieves documents using the original query, generates pseudo-documents, then re-applies retrieval
- *“Correct the retriever!”* (like adding annotations)

---

#### Maintaining performance when retrieval errors occur (Inference-based RAG)
**Noisy-Robust RAG**

⁉️ **What if the retrieval system is unstable?**  
(With only the question, the answer is correct—but when faulty documents are retrieved, the answer becomes wrong.)

**Training-Free Approach**
- Uses existing **Natural Language Inference (NLI) models**
- Achieves ~90% accuracy (SOTA) in identifying relationships between two sentences (entailment, neutral, contradiction)
- Filters documents by analyzing relationships between retrieved sentences and question–answer pairs
- Prevents performance drops, but **also limits the upper bound of performance gains**

**Small Training Approach (RetRobust)**
- Trains the model by showing both relevant and irrelevant documents and learning how to respond

---

## Improving RAG at the Training Level

**Expected benefits of RAG**
- Decide **whether to retrieve** given a query
- **Dynamically adjust responses** based on document quality

🧐 How can we achieve both effectively?

**Self-RAG**  
Define all desired behaviors manually, then fine-tune the model to realize them.  
❗ This makes the pipeline **complex**, since every desired behavior must be explicitly specified.

**Search-R1** (Reinforcement learning–based)  
**(Core idea inspired by) DeepSeek-R1**
- The model outputs *Reasoning* and *Answer* separately
- Assigns 1 point to the Answer and 2 bonus points to the Reasoning

**⇒ Introduces search-augmented tokens to solve RAG-specific problems**
- Adds *Search Call* tokens alongside Reasoning and Answer tokens  
- When such tokens appear, a search is triggered and results are inserted as *Search Result Tokens*
- The model is trained to generate increasingly accurate responses

**Reward function**: Only check final answer correctness!
$$
r_{\phi}(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})
$$

**Training strategy**: Evaluate only the parts generated by the LLM!  
(Inserted Search Result Tokens are excluded from evaluation.)

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x; \mathcal{R})} [r_{\phi}(x, y)] - \beta \mathbb{D}_{\text{KL}} [\pi_\theta(y \mid x; \mathcal{R}) \parallel \pi_{\text{ref}}(y \mid x; \mathcal{R})]
$$

**Overall Reinforcement Learning Algorithm**
$$
\begin{array}{l}
\textbf{Algorithm} \text{ LLM Response Rollout with Multi-Turn Search Engine Calls} \\
\hline
\textbf{Require: } \text{Input query } x, \text{ policy model } \pi_\theta, \text{ search engine } \mathcal{R}, \text{ maximum search budget } B. \\
\textbf{Ensure: } \text{Final response } y. \\
\quad 1: \text{Initialize rollout sequence } y \leftarrow \emptyset \\
\quad 2: \text{Initialize search call count } b \leftarrow 0 \\
\quad 3: \textbf{while } b < B \textbf{ do} \\
\quad 4: \quad \text{Generate response token } y_t \sim \pi_\theta(\cdot \mid x, y) \\
\quad 5: \quad \textit{// Append } y_t \textit{ to rollout sequence } y \\
\quad 6: \quad y \leftarrow y + y_t \\
\quad 7: \quad \textbf{if } \texttt{<search>} \dots \texttt{</search>} \text{ detected in } y_t \textbf{ then} \\
\quad 8: \quad \quad \textit{// Extract search query } q \\
\quad 9: \quad \quad q \leftarrow \text{Parse}(y_t, \texttt{<search>}, \texttt{</search>}) \\
\quad 10: \quad \quad \textit{// Retrieve search results} \\
\quad 11: \quad \quad d = \mathcal{R}(q) \\
\quad 12: \quad \quad \textit{// Insert } d \textit{ into } y \\
\quad 13: \quad \quad y \leftarrow y + \texttt{<information>}d\texttt{</information>} \\
\quad 14: \quad \quad \text{Increment search call count } b \leftarrow b + 1 \\
\quad 15: \quad \textbf{end if} \\
\quad 16: \quad \textbf{if } \texttt{<answer>} \dots \texttt{</answer>} \text{ detected in } y \textbf{ then} \\
\quad 17: \quad \quad \textit{// Terminate rollout} \\
\quad 18: \quad \quad \textbf{return } \text{final generated response } y \\
\quad 19: \quad \textbf{end if} \\
\quad 20: \textbf{end while} \\
\quad 21: \textbf{return } \text{final generated response } y \\
\hline
\end{array}
$$

⁉️ **학습되어 있지 않는 내용을 어떻게 LLM이 답변하게 할까?** 
⁉️ **새로운 지식을 계속 다시 알려주면 되지 않나?** => 기존 지식을 까먹는 현상 발생
## Retrieval System
#### Web Search
**Page Rank**: 중요도 = 하나의 웹사이트로의 링크를 포함한 다른 웹사이트들의 평균 중요도

$$
PR(A) = \frac{1 - d}{N} + d\left(\frac{PR(B)}{L(B)} + \frac{PR(C)}{L(C)} + \frac{PR(D)}{L(D)} + \cdots\right)
$$

$d$: damping factor : (1-d)는 연결된 페이지가 아닌 다른 임의의 페이지에서 넘어오는 경우
$PR(A)$: A에 있는 사용자가 해당 페이지로 넘어올 확률

⁉️ **질문의 맥락과 맞는 문서를 찾는다고 보장할 수 있을까?** No!

#### Text-base Retrieval
**BM25**(Raw Text-base Retrieval)
예) 겹치는 단어가 많을 수록 더 추출

$Q$: 질의 ($q_i$ 키워드 포함)
$N$: 총 문서의 개수, $n(q_i)$: $q_i$를 포함하는 문서 수

$$
\mathrm{score}(D,Q)=\sum_{i=1}^{n}\mathrm{IDF}(q_i)\cdot
\frac{f(q_i,D)\cdot (k_1+1)}{f(q_i,D)+k_1\cdot\left(1-b+b\cdot\frac{|D|}{\mathrm{avgdl}}\right)}
$$
$$
\mathrm{IDF}(q_i)=\ln\left(\frac{N-n(q_i)+0.5}{n(q_i)+0.5}+1\right)
$$

⁉️ **단어 하나로만은 문맥을 파악하기 어렵지 않을까?** 

#### Dense Retrieval
**Dense Passage Retrieval (DPR)**(Dense Retrieval)
두 개의 문장 인코더 사용 (예) BERT 모델
$$
sim(p,q)=E_Q(q)^TE_P(p)
$$
쿼리 $p$와 문단$q$의 유사성 = 두 벡터의 내적!

$\mathcal{D} = \{(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-)\}_{i=1}^{m}$
($p^+$: labeled data, $p^-$: labeled data for difficulty + in-batch negatives)

Constrastive loss:
$$
L(q_i, p_i^+, p_{i,1}^-, \ldots, p_{i,n}^-) = -\log \frac{e^{\mathrm{sim}(q_i, p_i^+)}}{e^{\mathrm{sim}(q_i, p_i^+)} + \sum_{j=1}^{n} e^{\mathrm{sim}(q_i, p_{i,j}^-)}}
$$
⁉️ **데이터를 만드는 비용이 너무 많이 든다!**
**Contriever**: 사람의 개입 없이 Dense Retrieval 모델을 학습할 수 있는 파이프라인
**Idea**: "*Wikipedia 같은 페이지에서 위쪽과 아래쪽은 양(+)의 관계가 있다.*" 
한쪽을 Query, 다른 한 쪽을 Positive Document라고 가정, 다른 페이지에서 온 부분을 Negative(-)로 가정

$k_+$: positive document (같은 문서에서 추출된 것)
$k_i$: negative documents (다른 문서)

$$
\mathcal{L}(q,k^+) = - \frac{\exp(s(q,k^+)/\tau)}{\exp(s(q,k^+)/\tau) + \sum_{i=1}^{K} \exp(s(q,k_i)/\tau)}
\qquad
s(q,d) = \langle f_\theta(q), f_\theta(d) \rangle
$$


## Improving RAG at Inference-level

#### 너무 많은 문서를 입력 받으면 어떻게 처리할 것인가?
**REPLUG (REtrieval and PLUG)**
**가정**: *LLM은 볼 수 없는 블랙박스다!* (모델 파라미터 등 모름)
⁉️ **LLM은 받아들일 수 있는 최대 문서 길이가 존재, 이를 넘어선다면 어떻게 할 것인가?**
문서를 한번에 다 보여주지 말고, **나눠서 여러가지 응답을 만든** 후, **응답 레벨에서 합치자**!
(문서의 중요도는 다르므로, 문서 중요도에 비례하게 답변에 가중치를 두어야 함)

$p(y \mid x, \mathcal{D}') = \sum_{d \in \mathcal{D}'} p(y \mid d \circ x) \cdot \lambda(d, x)$

**$\lambda(x,d)$: Similarity score from the used retrieval model**
$\lambda(d, x) = \frac{e^{s(d, x)}}{\sum_{d \in \mathcal{D}'} e^{s(d, x)}}$ (가중치 계산 (softmax))

⁉️**LLM, Retrieval은 학습이 되어있지 않기 때문에 성능 한계 존재**
(예) 4개의 문서 중 *실제로는 2번째 문서가 가장 중요도가 높지만*, REPLUG 이후 3번째 문서가 가장 중요도가 높다고 판단할 수 있음

**LSR Algorithm: 작은 Retrieval System을 LLM이 보는 관점으로 학습을 시키자!**
$P_R(d \mid x) = \frac{e^{s(d, x)/\gamma}}{\sum_{d \in \mathcal{D}'} e^{s(d, x)/\gamma}}$: Retrieval System의 확률분포
$Q(d \mid x, y) = \frac{e^{P_{LM}(y \mid d, x)/\beta}}{\sum_{d \in \mathcal{D}'} e^{P_{LM}(y \mid d, x)/\beta}}$: LLM의 확률분포
$\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} KL(P_R(d \mid x) \parallel Q_{\text{LM}}(d \mid x, y))$

#### Retrieval의 품질은 쿼리에 따라 달라진다! 더 좋은 쿼리 만들기
Retrieval이 학습 때 봤던 쿼리와 실제 쿼리 차이가 있을 때 성능 차이 발생 가능
🧐 LLM 모델 내부에 질문을 넣었을 때, **외부 의존 없이 질문과 관련된 문서를 생성**할 수 있다!

**HyDE**: 질문을 던진 후, GPT가 가상 응답 생성 이후 이를 Retrieval 시스템에 응답으로 제공
$sim(q, p) = \left( \sum_{k=1, \dots, K} E_P(\tilde{p}_k(q))/K \right)^T E_P(p)$
- **Contriever**  사용
- *"입력 쿼리를 변형!"*

**LAME-R** (Language Model Augmented Embeddings for Retrieval)
- **BM25** 사용 
- 쿼리를 가지고 기존 시스템으로 검색 후 가상문서 생성 -> 다시 Retrieval에 적용
- *"Retriever를 보정!"* (주석 붙여주듯이!)

#### 문서 검색 단계에서 에러가 발생했을 때 성능 유지하기 (Inference 기반 RAG)
**Noisy-Robust RAG**
⁉️ **Retrieval System이 불안정하다면??**
(질문만 줬을 땐 정답, Retrieval이 오작동해서 잘못된 문서 보여줄 때 잘못된 응답)

**Training Free Approach**
- 기존 **Natural Language Reference 모델** 사용
- 문장 두 개 주어졌을 때 두 문장간의 관계 (중립, 유사, 반대 등) 90% 달성 (SOTA 모델)
- 검색 문장 / 질문+응답 문장 이 둘 간의 관계를 파악하여 필터링 진행
- 성능이 떨어지는 걸 막지만, **성능이 높아지는 것의 상한값도 낮아진다.**

**Small Training Approach** (RetRobust)
- 관련 있는, 없는 문서 보여주고 어떻게 행동할 지를 학습시킴

## Improving RAG at Training-level

**<RAG에서 기대하는 효과>**
- Query 주어졌을 때 **검색 여부 결정**
- 문서의 질 기반 **응답 유동적 조정**

🧐 이 두 개를 효과적으로 달성하려면?

**Self-RAG**
원하는 행동들을 사람이 모두 정의한 이후, Fine Tuning 해서 이러한 성능을 구현하기
❗원하는 행동을 **일일이 지정해야 해서 파이프라인이 복잡**해진다!

**Search-R1**(강화학습 기반)
**(핵심 아이디어 기반) Deepseek-R1** 
	모델이 응답을 생성할 때, Reasoning과 Answer 부분을 나눠서 출력하도록 함
	Answer에 1점, Reasoning에 2점 가점 부여

**=> 검색 증강 생성에 특화된 토큰 도입 하여 문제 해결 시도**
- 기존 Reasoning, Answer 토큰에 Search Call 토큰, Call 토큰 도입하여 해당 토큰이 나올 경우, 이를 검색하여 Search Result Token으로 삽입
- 이후 정확도가 높은 응답을 계속 생성하도록 훈련

**보상함수** : 최종 정답 여부만 확인하자!
$r_{\phi}(x, y) = \text{EM}(a_{\text{pred}}, a_{\text{gold}})$

**학습방법** : LLM이 직접 생성한 부분만 평가하자! (추가된 Search Result Token은 평가에서 제외)
$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x; \mathcal{R})} [r_{\phi}(x, y)] - \beta \mathbb{D}_{\text{KL}} [\pi_\theta(y \mid x; \mathcal{R}) \parallel \pi_{\text{ref}}(y \mid x; \mathcal{R})]$

**전체 강화학습 알고리즘**
$$
\begin{array}{l}
\textbf{Algorithm} \text{LLM Response Rollout with Multi-Turn Search Engine Calls} \\
\hline
\textbf{Require: } \text{Input query } x, \text{ policy model } \pi_\theta, \text{ search engine } \mathcal{R}, \text{ maximum search budget } B. \\
\textbf{Ensure: } \text{Final response } y. \\
\quad 1: \text{Initialize rollout sequence } y \leftarrow \emptyset \\
\quad 2: \text{Initialize search call count } b \leftarrow 0 \\
\quad 3: \textbf{while } b < B \textbf{ do} \\
\quad 4: \quad \text{Generate response token } y_t \sim \pi_\theta(\cdot \mid x, y) \\
\quad 5: \quad \textit{// Append } y_t \textit{ to rollout sequence } y \\
\quad 6: \quad y \leftarrow y + y_t \\
\quad 7: \quad \textbf{if } \texttt{<search> } \dots \texttt{ </search>} \text{ detected in } y_t \textbf{ then} \\
\quad 8: \quad \quad \textit{// Extract search query } q \\
\quad 9: \quad \quad q \leftarrow \text{Parse}(y_t, \texttt{<search>}, \texttt{</search>}) \\
\quad 10: \quad \quad \textit{// Retrieve search results} \\
\quad 11: \quad \quad d = \mathcal{R}(q) \\
\quad 12: \quad \quad \textit{// Insert } d \textit{ into } y \\
\quad 13: \quad \quad y \leftarrow y + \texttt{<information>}d\texttt{</information>} \\
\quad 14: \quad \quad \text{Increment search call count } b \leftarrow b + 1 \\
\quad 15: \quad \textbf{end if} \\
\quad 16: \quad \textbf{if } \texttt{<answer> } \dots \texttt{ </answer>} \text{ detected in } y \textbf{ then} \\
\quad 17: \quad \quad \textit{// Terminate rollout} \\
\quad 18: \quad \quad \textbf{return } \text{final generated response } y \\
\quad 19: \quad \textbf{end if} \\
\quad 20: \textbf{end while} \\
\quad 21: \textbf{return } \text{final generated response } y \\
\hline
\end{array}
$$

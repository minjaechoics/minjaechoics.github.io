## 🔹 Greedy Decoding

**핵심 아이디어**  
가장 높은 확률의 토큰을 지속적으로 선택  

$x_{L+1}=arg$ $max$ $\hat{p}(x)$  

**장점**  
- Easy to use  

**단점**  
- It can be suboptimal depending on succeeding generations  


## 🔹 Beam Search

**핵심 아이디어**  
여러 가능한 후보 토큰에 대해 토큰 생성  

- 각 후보별 이후 여러 단계를 고려할 수 있음 (여러 분기 고려 가능)  
- 업데이트 시 토큰 선택 기준: (vanila) likelihood  
  $p(x)=p(x_1)p(x_2|x_1)...$  

**장점**  
- 더 좋은 결과를 생성할 수 있는 기회가 있음  

**단점**  
- 계산 비용이 높음  


## 🔹 Sampling

**핵심 아이디어**  
생성된 확률분포를 바탕으로 확률적으로 단어 선택  
(기존 Greedy Decoding, Beam Search는 확률이 높은 것을 *고정적으로* 선택)  

- LLM에게 같은 것을 물어봐도 다른 답변 얻을 수 있음  

**장점**  
- 토큰 생성 시 더 다양한 경우 가능  

**단점**  
- 토큰 생성 시 질적인 측면이 악화될 수 있음  

---

### ❓ 어떻게 Sampling Algorithm의 단점을 극복할 수 있을까?


### (1) Temperature  
의도적으로 모델의 행동 양식 조절  

예)  
$\hat{p}(x)=softmax(\hat{o}(x)/T)$  
// T가 크면 더 다양한 답변 생성 가능  


### (2) Top-K  
어차피 문제는 확률이 낮은 토큰들이니, 확률이 높은 k개의 토큰만 고려하자!  
=> 사용자 개입을 줄임  


### (3) Top-P (Nucleus Sampling)  
생성확률이 높은 토큰부터 하여,  
누적 확률 P가 Threshold를 넘지 않을 만큼의 토큰만 고려하자!  

- 고정된 숫자 ‘K’에 집중하지 말자  
- 👉 현재 주로 사용되는 알고리즘  

---

신약 개발 등 다양한 목적을 위해 LLM을 맞춤화할 때, LLM 전체를 다시 학습시키는 것은 비용이 너무 크다. 이에, 디코딩 알고리즘을 개선하여 산업 수요에 맞춘 LLM을 제작하는데 현재 연구되고 있는는 알고리즘들은 아래와 같다.

---

## 다양성 증대를 위한 Decoding Algorithm

### 🔸 Diverse Beam Search

**기존 방식**  
기존 Beam Search: Beam 개수를 확률적으로 가장 높은 K개 유지  

**문제점**  
확률분포만 집중하면 중복되는 토큰 수가 많아 다양성이 낮아진다!  

**개선**  
Beam을 선택할 때, 확률만 보지 말고, 이전에 만들었던 후보들과의 유사도를 패널티로 주자!  

$Y^{g}_{[t]}=argmax_{y^{g}_{1, [t]}, ..., y^{g}_{B', [t]}}{\varSigma}_{b\in[B']}\Theta(Y^g_{b,[t]})+\varSigma^{g-1}_{h=1}\lambda_g\Delta(y^g_{b,[t]},Y^h_{[t]})$  

$s.t.y^{g}_{i, [t]}\neq y^{g}_{j, [t]}, \lambda_g\leq0$  

---

## 모델 성능 자체 개선을 위한 Decoding Algorithm

### 🔸 Contrastive Decoding

기존 문제집: 큰 모델(Expert LLM)이 복잡한 문제를 푸는 과정에서 중간에 하는 작은 실수를 어떻게 줄일까?  

작은 모델(Amature LM)을 같이 써서 "Expert LLM 확률분포" - "Amature LM 확률분포" 로부터 도출된 확률분포에서 토큰 생성 시행 ($log{p_{EXP}}-log{p_{AMA}}$)  

#### ⚠️ Corner Cases 처리 방법

*큰 모델에선 생성되지 않은 토큰이 작은 모델에선 생성되면 어떻게 하지?* 등  

**Adaptive Plausibility Constraint**

(1) 큰 모델이 Rough 하게 토큰들을 필터링  
(2) 필터링 된 토큰들에 대해 작은 모델 활용  

$\mathrm{CD\text{-}score}(x_i; x_{<i}) = \log \frac{p_{\mathrm{EXP}}(x_i \mid x_{<i})}{p_{\mathrm{AMA}}(x_i \mid x_{<i})} \;\;\text{if } x_i \in \mathcal{V}_{\mathrm{head}}(x_{<i}), \text{ otherwise } -\infty$  


### 🔸 Visual Contrastive Decoding (VCD) (For Multimodal Models)

이미지를 토큰으로 바꾸어서 LLM이 처리하는 방식  

목표: Large Vision-Language Models (LVLMs)의 환각 현상 줄이기  

- Expert: 원본 이미지  
- Amature: 이미지에 노이즈를 크게 줌  

---

## 모델 속도 개선 Decoding Algorithm

### 🔸 Speculative Decoding

핵심 아이디어:  
토큰을 생성하다보면 생성하기 쉬운 위치가 있고, 어려운 위치가 있다!  

- e.g) The $blank$ : 어려운 케이스  
- e.g) The president was Barack $blank$ : 쉬운 케이스  


**(1) 작은 모델로 초안 작성 (여러 개의 토큰 한 번에 생성)**  
**(2) 생성된 토큰들을 큰 모델로 전달**  
: "*작은 모델이 만든 토큰을 큰 모델도 동의하는가??*"  
**(3) 적합하지 않은 토큰에 경우 큰 모델이 교정**  

❗ 한 번의 큰 모델 사용으로 여러개의 토큰 생성 가능  
❗ 추론 속도 역시 향상 가능  

# 16. Structured Probabilistic Models for Deep Learning

## Intro

- Structured probabilistic models: A modeling formalism
  - Graphical models(graph theory)
    - different models
    - training algorithms
    - inference algorithms

## 16.1 The Challenge of Unstructured Modeling

The goal of Deep learning

: Machine learning (->solve) AI challenges

### Tasks for Probabilistic Models

#### Classification algorithms

Rich high-dimensional distribution (->discard/ignore inputs) **single output**

#### More "expensive" tasks

- **Density estimation**:
  - x -> p(x): 1 output + Complete understanding of input
- **Denoising**:
  - incorrect/damaged x -> correct x: n output + Complete understanding of input
- **Missing value imputation**:
  - missing x -> entire p(x): n output + Complete understanding of input
- **Sampling**:
  - speech synthesis

##### Challenge

- Huge parameters:
  - **n discrete vars && k values each -> k^n parameters && O(k^n)**
    - reduce n -> dramatically reduce params
    - or: vars have fewer **parents** in the graph -> fewer params
- Memory
- Statistical efficiency
  - table-based model -> too many params -> overfit(unless back-off or smoothed n-gram models,etc)
- Runtime of inference and sampling
  - sampling: u~U(0,1) -> iterate adding until exceeding u -> return last added number

##### Problems

Truth:

- No need to modeling every possible kind of **interaction** between every possible **subset of variables**.
  - Most variables influence each other only **indirectly**.
  - e.g. A -> B -> C, when knowing B, C is only indirectly depends on A
    - modeling: A->B && B->C is enough, A->C can be omitted
- Structured probabilistic models -> only **direct interactions** -> fewer params -> less data

## 16.2 Using Graphs to Describe Model Structure

### Graphical models

Graph

: Nodes and Vertices connected by egdes

- Node: random variable
- Edge: direct interaction

#### 16.2.1 Directed Models(belief network or Bayesian network)

Directed graphical model

: Directed acyclic graph(DAG) G + Local conditional probability distributions $$p(x_i \mid Pa_G(x_i))$$

p(x):Joint Probability Distribution

: $$p(\mathbf{x}) = \prod_{i} p(x_i \mid Pa_G(x_i)).$$

##### Condition

**Understandable one-direction causality**(e.g. relay race)

##### Information can and cannot be encoded

- Simplifying assumption: vars are contionally independent
- Can: A+B
  - Only effect that time(A) has on time(B) is their summary A+B, not A->B
  - conditional distribution: from k * k - 1(table indexed) to k - 1
- Cannot: B is conditionally independent from A

#### 16.2.2 Undirected Models(Markov random fields (MRFs) or Markov networks)

No clean, undirectional narrative -> Edge isn't associated with a conditional probability distribution.

#### 16.2.3 The Partition Function

P775 - P512

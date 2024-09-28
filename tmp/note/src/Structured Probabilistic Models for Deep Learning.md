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

##### Normalized probability distribution

- $$p(x)=\frac{1}{Z}\tilde{p}(x)$$
- Z: An integral or sum over all possible joint assignments of the state x -> Hard to compute -> approximations
  - Z doesn't exist:
    - some vars are continuous and the integral of p over their domain diverges

##### Difference between directed/undirected modeling

776

#### 16.2.4 Energy-Based Models

One assumption: **∀x, p˜(x) > 0**  <- enforce this condition **EBM**

- $$\tilde{p}(x)=\exp(-E(x))$$
  - E(x): energy function
  - Probabilities always >0 -> we can use unconstrained optimization and don't need specific minimal probability value
  - Why **negative** E: preserve compatibility between the machine learning  literature and the physics literature.

##### Boltzmann machine

Many energy-based models use Boltzmann distribution -> **Boltzmann machines**

- Latent variables:
  - with: Boltzmann machine
  - without: Markov random fields or log-linear models

##### Cliques in an undirected graph -> factors of the unnormalized probability function

Product of experts

: an energy-based model with multiple terms in its energy function

- Single experts: low-dim projection of the random vars
- Mltiplication: complicated highdimensional constraint

##### Free energy

$$ F(x) = - \log \sum_{h} \exp{-E(x,h)}$$

#### 16.2.5 Separation and D-Separation

Separation / d(dependence)-Separation

: A is separated from B given S if the **graph** implies that A is independent from B given S

**? What is "Unobserved"?**

- Path between 2 vars:
  - Dependent/Separated vars/active path: path  involving only unobserved variables
  - Not separated vars / inactive path: no path exists between them, or all paths contain an observed variable

![alt text](image-4.png)

- Only **implied** conditional independences
  - Un-representable distribution: Context-specific independences

#### 16.2.6 Converting between Undirected and Directed Graphs

- Misleading: No probabilistic model is inherently directed or undirected.
  - Direct: draw samples from model
  - Undirect: deriving approximate inference  procedures
- Which approach capture the most independences/uses the fewest edges?
- Complete graph
  - d: each variable has all other variables that precede it in the ordering as its ancestors in the graph
  - ud: a single clique encompassing all of the variables

##### Specfic substructures

- Immorality(D)
  - (a,b) -> c, but a & b is independent("unmarried parents")
- Loop(U):
  - U contains a loop  of length greater than three, unless that loop also contains a chord
    - chord: connection between 2 non-consecutive vars in a loop
  - **U** -> Adding chords -> **chordal or triangulated graph**(discard some independence) + assign directions to edges and not create a directed cycle -> convert to **D**
    - tips: impose an ordering over the nodes(e.g. alphabetical order)![alt text](image-5.png)

#### 16.2.7 Factor Graphs

> The scope of every ^ function must be a subset of some clique in the graph.

- Nodes:
  - circles: random **variables**
  - squares: **factors** of the unnormalized probability distribution
    - Connected: Only when var is included in the arguments to the factor

- Why factors graphs are cheaper during representation, inference, and learning?
  - ![alt text](image-6.png)
  - > The computational complexity of each factor typically grows **exponentially** with the number of variables it is associated with.
  - n vars with k possible values: O(k^n)

## 16.3 Sampling from Graphical Models

### Ancestral sampling

- Topological ordering: arranging the nodes of a Directed Acyclic Graph (DAG) based on their **dependencies**(parent -> child, ensuring parent is available when sampling child)
- Ancestral sampling: sample following the causal order from the root nodes
  - adv: fast and convenient
  - disadv:
    - only applies to D
      - converting: trigger inference problems(determine the marginal distribution over the root nodes of the new directed graph)
        - e.g. A -> B -> C, sampling A when knowing B -> **P(A|B)**
      - UD: No clear begining point
    - **does not support every conditional sampling operation**

- Gibbs sampling(or MCMC):
  - Conditional independence: x_i is directly related only to its neighbors(iterable)
  - single pass is not sufficient -> iterate -> asymptotically converge
    - Difficulty: When is best?

## 16.4 Advantages of Structured Modeling

### Cost reduce(Primary adv)

- repre/infer/learning/sampling(accelerated)
  - How? -not to modeling all interactions

### Easier to develop / debug

- Separate representation of knowledge from learning of knowledge or inference given existing knowledge
  - Combine different algorithms and structure
  - Cartesian product of possibilities

#### end-to-end algorithms

less transferability/flexible/efficiency, more unexplainable

## 16.5 Learning about Dependencies

> A good generative model needs to accurately capture the distribution over the observed or “visible” variables v.

- Deep-learning: Introduce latent/hidden vars:h
  - v_i -> h -> v_j
  - No h: move parents per node(Bayesian network) or larger cliques in Markov n
  - Unable to connect all vars -> omit some edges
    - **Structure learning**
      - Greedy search: Rewards high acc and penalize model complexity -> add/remove egdes -> iter
    - Using h: avoid discrete searching and multiple rounds of training

### Latent variables

- Capturing p(v)
- Alternative representation of v
  - Mixtrue of Gaussians model -> Classification
  - Sparse coding -> input features for a classifier, or as coordinates along a manifold
  - feature learning
    - E(h|v): $$[ E[h | v] = \int h \cdot p(h | v) , dh ]$$
    - Argmax_h p(h,v): $$[ \text{argmax}_h  p(h, v) ]$$

## 16.6 Inference and Approximate Inference

Deep learning graphs: hard to inference

### #P-hard & Marginal Probability

- #P-hard: counting the number of solutions to a problem
- NP-hard
- Marginal Probability: the probability of a particular event occurring while ignoring the influence of other variables.
  - $$P(A)=\int P(A,B)\mathrm{d}B$$
- Reduction Tree:
  - Polynomial-time, transforming one problem into another

### Approximate inference

- variational inference: Use q(h|v) to approximate p(h|v) (U19)

## 16.7 The Deep Learning Approach to Structured Probabilistic Models

### Depth

Define: graphical model or computational graph

> a latent variable hi as being at depth j if the shortest path from hi to an observed variable is j steps.
> model depth: the greatest j

### Difference between Deep-learning and traditional modeling

#### Distributed representations

- Deep-learning: One single large layer of latent vars
  - more latent than observed vars
  - Complicated nonlinear interactions: indirect connections between multiple latent vars
    - Traditional models: through higher-order terms and structure learning

#### Designation of latent variables

- Deep-learning:
  - Do not take on any specific semantics ahead of time
    - hard for human to interpret after the fact
    - (more like end-to-end?)
  - Larger latent vars -> Need for efficient numerical code + high-level inference algorithm + group divided
- Traditional:
  - specific semantics & interpretable & **more theoretical guarantees**

#### Connectivity category

- Deep:
  - Connected large groups of units -> Overall Matrix
  - connect visible v to very many hidden h -> h can provide a distributed representation of v
    - Disadv of distributed representation: still not sparse enough for exact inference or lbp
      - -> dl almost never uses lbp
- Tra:
  - Few connections, individually designed
  - structure <-> inference algorithm
  - Typically aim: maintain the tractability of **exact inference**
    - **loopy belief propagation**

#### Unknown tolerance

> What's the minimum amount of information we absolutely need?
> How to get a reasonable approximation of that information as quickly as possible?

- Deep: High tolerance
  - marginal distributions cannot be computed -> draw approximate samples
  - intractable objective function -> efficiently obtain an estimate of the gradient of such a function

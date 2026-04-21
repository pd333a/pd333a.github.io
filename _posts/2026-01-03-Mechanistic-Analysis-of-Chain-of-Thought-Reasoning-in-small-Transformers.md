
In our daily life, when we provide mathematical simple problems  like “1+2=?” to a transformer, we see that it induces certain steps and provides answer as 3 but the question that arises here is that the intermediate steps that are a part of the inferencing are just some pattern-matching steps w.r.t some pre-trained material or are the steps directly influencing the model during the answer generation stage. In fact, if we go through Jason Wei’s chain-of-thought paper, we can observe that CoT[Chain-of-Thought] prompting definitely helps the model do better in reasoning based tasks but the above paper is an empirical one meaning it does not provide why it happens rather it defines performance improvements across multiple tasks . So, the below experimentation is an attempt to just see what happens under the hood of small transformer when it fed with CoT based samples inspired by some of the best works like  nostalgebraist’s gpt interpretation, Neel Nanda’s Grokking paper and special tutorial blogs.

First of all, what I feel is that mechanistic interpretability is a method to understand the internal algorithm that a LLM has learned after training it vasts amount of data. The main investigation that we will do in this project is to  validate certain claims/assumptions:

1.	Does the intermediate reasoning tokens used in Cot based prompting or finetuning a pre-trained model change the internal computations in the hidden states of the LL or is it just some form of pattern correlation between CoT based reasoning tokens and the answer in the prompt?
2.	Internal contribution of reasoning tokens towards answer generation in the attention phase and if any kind of causal ablation/corruption of the reasoning tokens directly influences the model accuracy

**SETUP**

In order to validate those assumptions, I  proceeded considering a small pre-trained LLM GPT2 with the simple  dataset of two-digit addition since it explicitly uses reasoning as its intermediate steps. Specifically speaking, I used HuggingFace’s GPT2LMHeadModel fine-tuned from the pretrained “gpt2” checkpoint. The model has 12 transformer layers, 12 attention heads per layer, 768-dimensional embeddings, and approximately 124M parameters. I used the standard GPT-2 BPE tokenizer. Both conditions are trained from the same pretrained checkpoint to ensure fair comparison. The reason for choosing the specific variant of LLM is that the LLM consists of comparatively smaller number of parameter and hence can be finetuned in a single gpu instance in the Google Colab. 

![llm_parameters](/assets/Screenshot 2026-04-22 003821.png)

The training data is generated synthetically in 2 formats : 1.CoT    and     2.Direct Answer

For CoT, we generated the data in string format consisting of explicit reasoning steps whereas the direct answer based data is consisting of only numeric computation results
E.g: CoT: "23 + 48 = 3+8=11, write 1 carry 1; 2+4+1=7 → 71"
Direct Answer: "23 + 48 = 71"

After this comes tokenization,which is the process of converting raw text into token ids and there is an embedding matrix for all such ids which can convert each id into an embedding vector in the vocabulary vector space storing various directional and positional information.

![Tokenizer](/assets/Screenshot 2026-04-22 014347.png)


The above code is important since the tokenize function returns a dictionary consisting of multiple keys like in  the below format

{
  "input_ids": [...],
  "attention_mask": [...],
  "labels": [...]
}

Where input_id is the target id for raw text and labels are used for auto-regressive training purpose and attention mask is used for padding token and text token classification

The next part was training. I trained the model with 10k such samples one on CoT based and the other model with same pre-trained checkpoint with direct-answer dataset with train and test split in the ratio 90:10. After training both the models and testing them on the test data, it was found expectedly that the CoT model has much higher accuracy than direct answer model.

After this I  considered the attention patterns. Specifically: when the model produces its final answer token, where is it looking?
I  categorizes every token in the sequence by its role — operand_a, operand_b, operator, reasoning step, or answer. Then I measured the average attention weight from the last token (the one producing the answer) to each category, averaged across all layers and heads and over 100 test examples.

The direct model distributes attention fairly evenly across the input operands. Makes sense — it has nothing else to look at. It's basically staring at "23" and "48" and trying to compute the answer internally. The CoT model directs the majority of its attention to the reasoning tokens. Not to the original operands — to the intermediate steps it generated. This implied that the model is most probably giving importance and utilizing the intermediate reasoning tokens.

**Logit Lens:**

It is the process of utilizing the hidden representations across the different layers of the transformer in identifying the most probable token the model is currently predicting.

The direct model keeps the correct answer probability low through most layers, then jumps at the final layers. It's doing most of its computation at the end — a compressed, concentrated effort.

The CoT model builds the answer gradually. Probability increases steadily from the embedding layer through to the final layer. Each layer contributes a little more toward the correct answer.

This is a qualitative difference in computational structure. The CoT model distributes its computation across layers. The direct model concentrates it. CoT doesn't just add tokens — it changes how the network processes information.

Now, I can make an observation on the basis of the behaviour of the direct answer model. From Neel Nanda’s grokking paper, we can see that grokking is a phenomenon in which the model initially at first memorizes training data, then after long training suddenly generalizes perfectly when there is a sudden phase transition after which internal representations reorganize into cleaner algorithmic circuits. So, from the above experiment, we can say that CoT based training triggers fast revival of grokking behaviour in models in comparison to the direct answer based training in the models.

**Causal experiments:**

These experiments were done to determine the influence of the reasoning tokens on the model’s own behaviour. 
Experiment 1: Full removal. I took CoT test sequences and deleted all reasoning tokens, converting "23 + 48 = 3+8=11, write 1 carry 1; 2+4+1=7 → 71" into just "23 + 48 = 71". Then I measured accuracy.

The accuracy dropped by roughly 65%.
This implied that the model is completely dependent upon the reasoning tokens and is using those as reference to compute the final answer.

Experiment 2: Shuffling. I kept all the same reasoning tokens but randomized their order. If the model just needs the presence of reasoning-like tokens, shuffling shouldn't matter. If it uses the sequence of steps, shuffling should impact accuracy.
Accuracy dropped significantly — less than full removal, but substantially. This implied two things: the model uses the order of reasoning steps (not just their presence), and some information survives shuffling (token identity matters independently of position).

Experiment 3: Text corruption. I replaced increasing fractions of the digits in the reasoning portion with random digits. So "3+8=11" might become "7+2=15" — wrong numbers, but same format.

The accuracy degraded smoothly with corruption level. It was like a slope. This implies that the model is not memorizing the results and storing them in a table. If the model stored reasoning as a fragile template, even small corruption would catastrophically break it. Smooth degradation means the information is distributed across the reasoning representations.
Here,we can make a specific scenario that supports the above. A feature is a direction in the activation space that corresponds to an interpretable concept and a specific token might be present at the intersection of multiple conceptual features and thereby enforces the superposition phenomenon where the neural network encodes more features than it has dimension into non-orthogonal directions and hence information is distributed across multiple directions, corruption causes gradual rather than sudden degradation.

Experiment 4: Partial ablation. Here instead of removing all reasoning, I removed individual sub-steps one at a time:
•	Remove just the carry step ("carry 1")
•	Remove just the ones addition ("3+8=11")
•	Remove just the tens addition ("2+4+1=7")
The carry step caused the largest accuracy drop when removed. By a good margin. This makes complete mechanistic sense: the carry is the one piece of information that must flow from the ones computation to the tens computation. It's the inter-column dependency. Without it, the model has to figure out the carry implicitly, and it can't do that reliably.

**Some observations that we can make out of this experiment:**
There are 2 scenarios that we have to consider here: 1. There is a dependency on intermediate reasoning tokens that we can confirm from the causal experiment and also we have to consider the fact that 2. There is also some moderate level accuracy associated with the direct model which we also have to consider since in that case it is not taking help of those reasoning tokens.

From the above considerations, we can say that the model is somewhat behaving as a mixed/hybrid way.It might be using 2 types of channels. In one channel, it reads the reasoning tokens sequentially, uses the carry information, follows the step order. Also, there can be a another latent channel where the model performs computations in the residual stream without considering the intermediate reasoning tokens when done in the case of direct-answer training.

**Some important stuffs that can be tried upon:**

For this experiment, we used an already pre-trained model which was initially having prior knowledge in numbers and also we used a smaller scale model to facilitate faster model training through single gpu. So, we would like to train a large scale model without any pre-training in order to capture whether the model develops any raw internal algorithm or does it develop any specialized circuit that enables efficient reasoning in multiple such tasks.



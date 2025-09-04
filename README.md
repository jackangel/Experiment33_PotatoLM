# PotatoLM - "Fake" Attention is all you need

Tired of quadratic complexity? Annoyed by attention matrices that are bigger than your hard drive? Do you look at a softmax and think, "there has to be a simpler way"?

Well, there probably is, but this is what we came up with instead. Welcome to the architecture of PotatoLM, a model built on the groundbreaking principle of just passing information along in a line, but, you know, *in parallel*.

---

### The Core Idea: "Fake Attention"

The heart, soul, and starchy center of this model is the `ChunkedMultiHeadCardPassingLayer`, which we affectionately call **Fake Attention**.

Why? Because it's a desperate attempt to get the benefits of sequential information flow (like an RNN) while keeping the parallel processing goodness of a Transformer. It's not really attention, but if you squint and turn your head, it kinda serves the same purpose: letting tokens talk to each other.

The core metaphor is a game of "pass the card". Imagine your input tokens are sitting in a long row.
1.  Each token thinks of a secret message to write down (a "mark").
2.  It also decides how loudly it wants to write this message (a "gate").
3.  These messages are collected on a "card" that gets passed down the line, so token `T` gets to read the combined messages of all tokens from `1` to `T-1`.

This sounds terribly slow and sequential, right? Wrong! We cheat.

### The Chunking Miracle: How to Pass Cards in Parallel

This is where the magic happens. Instead of passing the card one token at a time, we break the sequence into smaller, manageable `chunks`. The information passing now happens in two glorious, parallelizable stages.

#### Stage 1: Local Gossip (The Intra-Chunk Scan)

Within each small chunk, all the tokens share their secrets simultaneously. We use a parallel `cumsum` operation to instantly calculate the cumulative "card" at every position *within the chunk*. Everyone in the group immediately knows the combined secrets of all their local neighbors who came before them. It's like speed-reading the room's gossip.

#### Stage 2: The Town Crier (The Inter-Chunk Scan)

Now we need to get information *between* the chunks.
1.  We take the total sum of all secrets from each chunk (the "gossip summary").
2.  We then perform another parallel `cumsum` on these *summaries*.
3.  This gives us a "carry" value for each chunk, which represents the combined knowledge of *all previous chunks*. It's like a town crier running to the start of each chunk and shouting a summary of what happened in the previous towns.

#### Stage 3: The Grand Combination

Finally, every token gets its final, wisdom-filled card. This card is the sum of:
1.  The local gossip from inside its own chunk.
2.  The "town crier" news from all preceding chunks.

The token then looks at its original self, looks at the rich history of context written on the card it just received, and produces its output. This output is then passed to the next Fake Attention layer, and the whole ridiculous process starts over again.

### What About The Rest of the Model?

Honestly, it's pretty standard.
*   We stack a bunch of these `Fake Attention` layers on top of each other.
*   We even splurged on **Rotary Positional Embeddings (RoPE)** because even a potato needs to know where it is in the sack. It gives our tokens a sense of direction before they start passing cards all over the place.
*   Then there's a final LayerNorm and a linear head to predict the next token. Simple.

The entire architecture is an experiment in replacing the expensive, quadratic attention mechanism with a linear, parallel-scan-based alternative that tries its very best to remember things.

---

Can you have a 3M parameter LM?

The answer is - a resounding - yes, maybe, I don't know man. Try it for yourself, it won't cost any money.

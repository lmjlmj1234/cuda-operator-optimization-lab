# Problem Definition

**What:** Softmax normalizes an input vector into a probability distribution:
`softmax(x)_i = exp(x_i) / sum_j exp(x_j)`

**Why:** Fundamental building block in Transformer attention layers (after QK^T scores), classification heads, and any multi-class output layer.

**Role in Transformer:** The normalization step in `Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V`. Must be numerically stable and efficient for large sequence lengths and hidden dimensions.

**I/O Shapes:** `N=4096` rows, `D=8192` columns, float32.

**Complexity:** O(N*D) reads, O(N*D) writes -- 3x HBM reads for naive (read input twice + read max/sum).

**Memory Pattern:** Row-wise: each row is contiguous in memory (coalesced), each block handles one row.

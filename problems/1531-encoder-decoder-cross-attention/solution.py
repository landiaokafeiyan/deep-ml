import numpy as np

def cross_attention(
    decoder_x,
    encoder_out,
    W_q,
    W_k,
    W_v,
    src_pad_mask=None
):
    """
    Encoder-decoder cross attention.

    Args:
        decoder_x:
            shape (tgt_len, d_model)

        encoder_out:
            shape (src_len, d_model)

        W_q:
            shape (d_model, d_k)

        W_k:
            shape (d_model, d_k)

        W_v:
            shape (d_model, d_v)

        src_pad_mask:
            shape (src_len,)
            True / 1 means padded source position

    Returns:
        output:
            shape (tgt_len, d_v)

        weights:
            shape (tgt_len, src_len)
    """

    # -------------------------
    # 1. Linear projections
    # -------------------------
    Q = decoder_x @ W_q
    K = encoder_out @ W_k
    V = encoder_out @ W_v

    # Q: (tgt_len, d_k)
    # K: (src_len, d_k)
    # V: (src_len, d_v)

    d_k = Q.shape[-1]

    # -------------------------
    # 2. Attention scores
    # -------------------------
    scores = (Q @ K.T) / np.sqrt(d_k)

    # scores:
    # (tgt_len, src_len)

    # -------------------------
    # 3. Apply source padding mask
    # -------------------------
    if src_pad_mask is not None:
        mask = np.asarray(src_pad_mask).astype(bool)

        # Mask source columns
        scores[:, mask] = -np.inf

    # -------------------------
    # 4. Stable softmax
    # over source axis
    # -------------------------
    row_max = np.max(
        scores,
        axis=1,
        keepdims=True
    )

    shifted_scores = scores - row_max

    exp_scores = np.exp(shifted_scores)

    weights = (
        exp_scores
        /
        np.sum(
            exp_scores,
            axis=1,
            keepdims=True
        )
    )

    # -------------------------
    # 5. Weighted sum of V
    # -------------------------
    output = weights @ V

    # output:
    # (tgt_len, d_v)

    return output, weights
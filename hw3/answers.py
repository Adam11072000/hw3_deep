r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=500,
        seq_len=300,
        h_dim=512,
        n_layers=3,
        dropout=0.5,
        learn_rate=0.001,
        lr_sched_factor=0.7,
        lr_sched_patience=5,
    )
    return hypers


def part1_generation_params():
    start_seq = "ACT I"
    temperature = 0.4
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
we split the text into sequences for several reasons:
1. parallilism of our training thus faster training.
2. back propagation on large text might lead to vanishing/exploding gradients, thus training will be impossible.
"""

part1_q2 = r"""
**Your answer:**
our model can remember with no limit to the "past", due to the fact that the update and reset gate decide how much
the next state is affected by the previous state.
"""

part1_q3 = r"""
**Your answer:**
because we want to retain the semantics and meaning between sequences and have a 
more logical and plausable output of the model.
"""

part1_q4 = r"""
**Your answer:**
1) we lower the temperature in order to shift the distribution to a more heterogenous distribution, thus making the generated sequences more 
related to what is learnt.

2) when the temp is very high, the distribution will be more uniform, thus sampling from the distribution will yeild random input.

3) when the temp is very low, hot-softmax will behave like argmax, thus the letter with the highest distribution will be sampled. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 128, 
        num_heads = 4,
        num_layers = 3,
        hidden_dim = 192,
        window_size = 128,
        droupout = 0.20,
        lr=0.0004,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
stacking multiple layers in a CNN resuls in that each layer processes an input that was processed by the previous layer, thus each layer increases the total receptive field of the input.
in a sliding window attention, each layer processes an input from a larger context that was captured by the previous layer, thus increasing the receptive field
"""

part3_q2 = r"""
**Your answer:**
a trivial answer is to make the window bigger, thus having a more global context than a small window (check with abugosh)
"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============

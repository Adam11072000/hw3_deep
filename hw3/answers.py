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
    hypers = dict(
        batch_size = 64,
        h_dim = 256,
        z_dim = 64,
        x_sigma2 = 0.0015,
        learn_rate = 0.0003,
        betas = (0.9, 0.9)
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The sigma squared is a hyperparameter which basically controls variance from the mean image.
For low values of sigma, the model will generate photos that are closer to the mean image that the model has learnt thourghout the training process.
For higher values of sigma, the model will branch out and generate images that can vary from the mean image.

"""

part2_q2 = r"""
**Your answer:**
1. reconstruction loss: gives an indication of how good/bad the decoder tried to reconstruct the original signal x.
   KL divergence: gives an indication of the difference between the learnt latent distribution of the VAE and the original set distribution.
   
2. Basically the KL divergence loss changes the mu and sigma_2 parameters to make the posterior distibution of the latent close as possible to the prior distribution.

3. The main benefit of using KL Divergence loss term is that making the posterior distibution of the latent close as possible to the prior distribution will gives images that are closer to 
   the original picture, that is due to the fact that we trained our model to generate images that are in the vacinity of the original image.
"""

part2_q3 = r"""
**Your answer:**
We maximize the evidence distribution to inderictly maximize the observed data probibality, which in turn ensures that we can find the distribution of where the learnt data came from.


"""

part2_q4 = r"""
**Your answer:**
There are a couple of reasons of why we may use the log space:
1. Log treats us with significantlly smaller numbers than regular space, this may ensure that we are more stable when performing mathematical operations.
2. P(x) is a multiple between P(x_i) for i in data, we saw in machine learining course before that it is easier to solve an Minimization/Maximization problem in the log scale than the regular scale because we will work on sum rather than multiplication

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
one way is dialated window attention, where the model attends to the current window w, can also attend to other elements in the sequence that are exponentially far from the curent position in the sequence.
for example, if the model is attending to position i and its window, the model can also take into account i + 2w, i + 4w, i + 8w, .... elements and attend to them, but with smaller weights becauser intuitivly, more attention should
be given to closer positions rather than distant elements.
the number of elements that the model attends to can still be capped to ensure the same time complexity.
"""


part4_q1 = r"""
**Your answer:**
in comparission to part3, the fine tuning performed better without any outside factors. 
The reason may be credited to multiple factors:
1. The model had a noticeably larger dataset for the training.
2. The pretraining tasks were quite similar to the downstream tasks.
3. The model itself we are using is larger than the model used in the last part.
from reason 2 we can conclude that the pretraining phase was a very helpful way to ensure we get pretty good results.
To summarize, if the downstream task is within the same domain/realm of the pretrain tasks, we expect to see better results in fine tuning over trainig a model from scratch, however, if the downstream task's realm is unknown or the different from the pretrain tasks, we may find that the scratch model will give better results.
"""

part4_q2 = r"""
**Your answer:**
The following suggestion (internal layer freezing) is not ruled out for fine tuning purposes, however, if we look into it from a general point of view, it is likely that the results may be worse that fine tuning the later stages of layers.
Per defintion, the earlier layers (beginning and middle) grasp a more general idea of our features while the later stages are those who are task specific.
This in turn means that freezing a middle layer can majorly distrubt the training proccess and may interfere with knowing how to similar features/sequences that are found in different tasks.

"""





# ==============

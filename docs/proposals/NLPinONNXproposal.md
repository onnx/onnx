<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

## Background

Modern NLP (Natural Language Processing) is an important domain in which deep learning is applied. In addition, modern NLP networks are often non-trivial to implement and even more difficult to transfer between frameworks. These networks are handled fairly non-uniformly across the landscape of frameworks. The ability for ONNX to interchange these networks can be a very compelling feature.

NLP networks, including recurrent networks, are often built on dynamic control structures. Standardizing the handling of these structures can lead to better collaboration with backends to expose network semantics and achieve better performance. A tradition has developed within the Computer Vision field for optimizing hardware backends for canonical vision models, such as ResNet-50. There is not really such as tradition in the NLP field, however. Through standardizing the representation of NLP networks, we can give vendors a common representation and push forward the performance of NLP models.

## Ultimate Goal and Challenges

We should work toward being able to represent major classes of NLP model architectures. One example of such an architecture is the seq2seq with attention model (e.g. https://arxiv.org/abs/1409.0473). This architecture is used for many use cases, including neural machine translation, speech processing, summarization, dialog systems, image captioning, and syntax parsing, among many others. At the same time, seq2seq with attention is sufficiently complex that supporting it will push forward the state of the art in ONNX, but not so complex that we'd need to define a full programming language.

seq2seq with attention can roughly be broken down into these constituent parts:

* An Encoder network
    * This network takes a sequence of tokens and yields a sequence of embeddings representing the context found at each time-step
    * Major classes of encoders: recurrent network (e.g. LSTM[1]), convolutional[2], attention[3].
    * Requirements from an ONNX representation
        * Recurrent network - general recurrent network structures preserving outputs at every timestep. Handling of padding and hidden states for batches with different sequence lengths).
        * Convolutional - 1d convolution, position embeddings
        * Attention - sinusoid position encodings, layer normalization
* A Decoder network
    * This network generates a sequence token by token, parameterized by the context provided from the encoder.
    * Yields a probability distribution over possible tokens given previous context and encoder context.
    * Major classes of decoders: recurrent network (e.g. LSTM), convolutional (causal, temporal for generation), attention.
    * Generation requires dynamic control flow. Often, this is done as a beam search, so this is distinct from regular recurrent networks.
    * Model-specific requirements
        * Recurrent network - Recurrent network cell that can be used within the context of beam search
        * Convolutional - 1d causal convolution (only see previous timesteps)
        * Attention - sinusoid position encodings, masking along diagonal
* An Attention mechanism
    * This network weights the Encoder contexts based on the Decoder's generation state, and provides a focused Encoder context to the decoder. The Decoder “focuses” on a certain part of the input sequence at each timestep via this mechanism.
    * Many classes of attention mechanism: some examples are here https://arxiv.org/pdf/1508.04025.pdf


Vanilla seq2seq with attention and non-backtracking beam search does NOT include things such as auxiliary data-structures (e.g. stacks), thus it does not require us to implement the full semantics of a programming language. It is an architecture that we can break down into incremental improvements to ONNX without compromising ONNX's fundamental goal.

[1] https://arxiv.org/abs/1409.0473
[2] https://arxiv.org/abs/1705.03122
[3] https://arxiv.org/abs/1706.03762

## Standard Recurrent Network Constructs

Standard recurrent network architectures such as LSTM or GRU are very common, and we can get very far supporting these. We already have the [LSTM](/docs/Operators.md#LSTM) and [GRU](/docs/Operators.md#GRU) operators, which execute the standard LSTM and GRU[4] operations over a sequence of inputs. These high-level operators are great, since they give backends a semantic view of the computation to be performed, and thus backends can make informed decisions about optimization. Many NLP use cases can get away with using just these operators.

[4] http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Generic Control Flow

Once we move beyond the domain of standard LSTM and GRU operations, we need a more generic abstraction onto which we can map NLP architectures. A simple example is how one can implement Multiplicative Integration LSTM (https://arxiv.org/pdf/1606.06630.pdf) in ONNX. We can expose a standard LSTMCell via the proposed Function abstraction (https://github.com/onnx/onnx/issues/481). Building on top of this, we can construct a MI-LSTM by applying the required second-order transformations to the inputs to the LSTMCell. Once we have this aggregated implementation, we can use the generic control flow operators (https://github.com/onnx/onnx/pull/436) to apply this “composite” MI-LSTM cell over a sequence.

Of course, the dynamic control flow constructs can be used for more general use cases. For example, consider the [beam search](https://en.wikipedia.org/wiki/Beam_search) used often in NLP for sequence generation. This algorithm has several tricky aspects: a (potentially) dynamic stopping condition, a desired maximum trip count (so we don't fall into an infinite loop), loop-carried dependencies, and the desire to preserve the outputs at every time-step, not just the final time-step. Inherently, this is an imperative algorithm that operates on mutable state. The proposed control flow operators in ONNX, however, fulfill all of these requirements, and thus we can represent many instances of sequence generation in ONNX graphs.

Note that there are more general forms of beam search, such as those including backtracking, but we are not considering these forms for this focused proposal.


## End-to-end Example : seq2seq with attention

We should endeavor to have full support for seq2seq with attention models in ONNX. Facebook is currently working on this internally and creating a pytorch→ONNX→caffe2 pathway. An example of such a model we'd like to represent in ONNX is [fairseq](https://github.com/facebookresearch/fairseq). We would love to engage with the community and collaborate on anything that will help make this a reality. Additionally, if the community has any other suggestions for prominent NLP models we should be able to represent, we would love to hear your ideas.

## Further Challenges

Beyond the constructs used in seq2seq with attention, there are NLP models that exist today that contain more non-trivial features, such as mutable data structures that are manipulated at runtime. Examples of this include back-tracking beam search and parser models such as RNNG (https://arxiv.org/abs/1602.07776). These will present further challenges for ONNX, and the representation of these models will likely remain tied to application code for the time being. We may want to revisit this class of models in the future.

Another thing we should consider is how to handle preprocessing and postprocessing routines for NLP models. For example, do we defer tokenization, normalization, and index lookup to application code? And how do we, for example, distribute dictionaries that map tokens to indices. Initially this will probably remain out of the scope of ONNX unless there is a good story for standardizing text processing.

## Conclusion

We have presented a proposal for a strategy for representing NLP models in ONNX, using seq2seq with attention as a canonical example that covers many use cases. We would like to hear your thoughts about this proposal and to explore opportunities for collaboration with the ONNX community for making ONNX a pleasure to use for NLP. Please feel free to voice your opinions!

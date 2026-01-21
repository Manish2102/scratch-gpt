hey all this is a project that helps you in building a gpt style model from scratch

take a dataset in this project i have taken a text data from kaggle

first we perform tokenizer

in this step we normalize text and we can either spilt the text to words or characters

after that we convert character or words to numbers i.e token id

we defined the vocabulary in vocabulary.json file which has numbers that are assigned to each character and special characters 

based on the vocabulary we can convert the text to token id



First: What GPT Is Actually Learning

Forget “AI”, “language”, and “transformers” for a moment.

GPT learns exactly ONE task:

Given some tokens, predict the next token.

That’s it.
No more. No less.

What You Have Without Training Sequences

Right now, after tokenization and encoding, you have this:

[41, 70, 77, 77, 80, 1, 88, 80, 83, 77, 69, ...]


This is just a long list of numbers.

Ask yourself:

Where is the question?

Where is the answer?

What is the model supposed to predict?

👉 There is no learning signal yet.

training sequences solve 
neural networks do not learn from raw data they learn from input -->  expected output pairs 

A training sequence is:

A fixed-length window of tokens (input),
and the same window shifted by one token (target).

you define the block size and give the input as tokens and block size

we define block size so that model can look at the last 8 tokens and predict the next one 

Training sequences let GPT:

learn from raw text

without labels

without annotations

without human supervision

The text labels itself.

This is called self-supervised learning


What the NEXT STEP Will Produce

After the next step, you’ll have:

x_tensor → shape: (block_size,)
y_tensor → shape: (block_size,)
dtype: long (int64)


This is exactly what embeddings expect.

🔜 WHAT COMES AFTER THAT (Preview Only)

positional embeddings 

after token embeddings we will add positional embeddings
the model doesnt know the order of the tokens which we resolve using positional embeddings
 
the positional embeddings are given each a learnable vector 

for sequence of length 8, we will have 8 positional embeddings

Gpt does not conatenate it adds final_embedding = token_embeddings + positional_embeddings

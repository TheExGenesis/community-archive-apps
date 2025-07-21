# Notes about plotting social fabric

2025
jul 11

- Just came to mind that it may be a lot easier to make sense of only top level tweets without pictures
- Most tweets are reactions - I wonder if we could filter for "high content" tweets
- ok I made a function to get the top tweets for each user, and cleaned them up a bit. They encode some information, but ideally we'd get an llm description of the user's content. I think for now I'll just go with this to try the covariate embeddings. , and it's a react package

jul 10

- brandon suggested clustering after UMAP, not before, if you're trying to plot clusters. Seems counterintuitive to me but it's def more efficient and it seems like people do it. https://umap-learn.readthedocs.io/en/latest/clustering.html
- got embedding and clustering working from a modal function - test doesn't cluster as expected but I only have about 5 points
- ok I embedded all the tweets for a week, am clustering them on 10d embeddings - can only get a bit above 50% non-noise points at a reasonable number of clusters, 100+
- Still haven't made the user vectors, I seem to have temporarily redirected to just plotting and clustering tweets. What's the purpose? I'm not sure, mostly exploratory for now. If I were trying to pick up on trends, I wonder if I'd do stuff like filter for top level posts and filter for high likes / retweets.

next:

- [ ] try datamapplot on the tweets use top tfidf as labels, then maybe even try llm to generate labels
- [ ] use week's tweets to generate a summary of each user, then embed it, it's fine - it's just to add a bit of semantic information as covariates to the network embeddings

jul 9

- Datashader is cool but making the interactivity out of the box makes it super slow. I wonder if I can do something better later
- Bundling looks cool but it's sometimes tricky to understand connections, sometimes far away nodes are connected, sometimes nodes that seem to be part of different branches are also connected.
- Datashader rerenders on when served by a server but not when served by a notebook.
- Spectral embeddings look very weird with power law distributions, look very thin, degree dominates a lot, even with Laplacian. Similarity matrix also looks quite collapsed.
- node2vec as a layout looks good and users that are nearby make some sense sometimes, but not always, isn't amazing
- Now I'm wondering about doing covariate assisted embedding, using like embeddings of summaries of people's content as covariates.
- Also wondering abvout multiple adjacency matrices, like followers and interactions, and likes, and doing MASE embedding on those.

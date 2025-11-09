Help me proofread The challenge was to create a multimodal search system to search through HEB's product catalog and get the desired products. 

We decided to approach this by using RAG techniques, combining the results from different models (dense vectors, sparse vectors, and clip vectors from a shared vector space for images), and trying out different combinations of these vectors to rank results. We also applied reranking techniques to all of the results to capture more semantic meaning. We also attempted finetuning the sentencetransformers model, however experienced deteriorating performance and returned to the non-finetuned version.

We used sentence transformers' model for dense vectors, ngram encodings for sparse vectors, and clip for image/text space vectors. We used the bge-reranker models from huggingface for reranking, and queried them through the huggingface api. We thought a finetuned version of sentence transformers would make for good semantic meaning capture, but didn't see the results we wanted. However, the basic sentence transformers worked well. We also used ngram encodings due to Julio's previous experience with them creating RAG systems in the manufacturing industry. We thought it would capture specific ingredients and other multi-word patterns well. Finally we used clip for the image/text vectors, clip models are great for this kind of task where we want to use text to search through images.

Our tech stack was Java SpringBoot for the orchestration layer, which handled most of the workflow, and Python for serving and finetuning models. Python models were served through FastAPI endpoints which made it very easy to host the embeddings models locally. We also used a python script to evaluate all queries, as the java backend was designed to handle single query requests (like we would expect in the real world).

Performance increased the most when we added reranking, especially nDGC@10. Adding ngram and image embeddings improved the R@30 score the most, seemingly indicating that searching through vectors was the best way to get initial results, and reranking was the best way to sort them correclty. We avoided more brute force solutions like having a model manually rank the products because that would defeat the purpose of the challenge, using vector search and reranking is actually reasonable.




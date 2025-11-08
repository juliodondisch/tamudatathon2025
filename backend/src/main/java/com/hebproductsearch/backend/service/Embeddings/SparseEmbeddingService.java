package com.hebproductsearch.backend.service.Embeddings;

import java.util.ArrayList;

import org.springframework.stereotype.Service;

@Service
public class SparseEmbeddingService {
    public ArrayList<Float> getEmbedding(String info){
        // Function will query ngram encoder backend
        ArrayList<Float> embedding = new ArrayList<Float>(384);
        for (int i = 0; i < 1000; i++) {
            embedding.add(1.0f);
        }
        return embedding;
    }
}

package com.hebproductsearch.backend.service.Embeddings;

import java.util.ArrayList;

import org.springframework.stereotype.Service;

@Service
public class DenseEmbeddingService {
    
    public ArrayList<Float> getEmbedding(String info){
        // Function will query dense embedding backend.
        // temporarily return an arraylist of 384 1s
        ArrayList<Float> embedding = new ArrayList<Float>(384);
        for (int i = 0; i < 384; i++) {
            embedding.add(1.0f);
        }
        return embedding;
    }
}

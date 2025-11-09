package com.hebproductsearch.backend.service.Embeddings;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class SparseEmbeddingService {

    @Value("${sparseembedding.url}")
    private String embeddingUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public ArrayList<Float> getEmbedding(String info){
        // JSON request body
        Map<String, String> requestBody = Map.of("query", info);

        // Make the POST request and parse response as Map
        Map<String, Object> response = restTemplate.postForObject(
            embeddingUrl,
            requestBody,
            Map.class
        );

        // Extract the embedding array from JSON
        List<Double> returned = (List<Double>) response.get("sparse_embedding");
        // Convert double list → float list
        ArrayList<Float> embedding = new ArrayList<>(returned.size());
        for (Double val : returned) {
            embedding.add(val.floatValue());
        }

        return embedding;
    }
}

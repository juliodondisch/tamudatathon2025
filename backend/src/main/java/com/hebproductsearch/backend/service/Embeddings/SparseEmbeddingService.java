package com.hebproductsearch.backend.service.Embeddings;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class SparseEmbeddingService {

    private final RestTemplate restTemplate = new RestTemplate();
    private static final String PYTHON_SERVICE_URL = "http://localhost:8001/sparse-embed";

    public ArrayList<Float> getEmbedding(String info){
        try {
            // Create request body
            Map<String, String> requestBody = Map.of("query", info);

            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            // Create HTTP entity
            HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

            // Make POST request to Python service
            ResponseEntity<Map> response = restTemplate.postForEntity(
                PYTHON_SERVICE_URL,
                request,
                Map.class
            );

            // Extract embedding from response
            if (response.getBody() != null && response.getBody().containsKey("sparse_embedding")) {
                List<Double> embeddingList = (List<Double>) response.getBody().get("sparse_embedding");
                ArrayList<Float> embedding = new ArrayList<>(embeddingList.size());

                for (Double value : embeddingList) {
                    embedding.add(value.floatValue());
                }

                log.info("Successfully got sparse embedding of dimension: {}", embedding.size());
                return embedding;
            } else {
                log.error("No sparse_embedding in response");
                return getFallbackEmbedding();
            }

        } catch (Exception e) {
            log.error("Failed to get sparse embedding from Python service: {}", e.getMessage());
            return getFallbackEmbedding();
        }
    }

    // Fallback to dummy embedding if Python service is unavailable
    private ArrayList<Float> getFallbackEmbedding() {
        log.warn("Using fallback embedding (all zeros)");
        ArrayList<Float> embedding = new ArrayList<Float>(1000);
        for (int i = 0; i < 1000; i++) {
            embedding.add(0.0f);
        }
        return embedding;
    }
}

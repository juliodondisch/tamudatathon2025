package com.hebproductsearch.backend.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.hebproductsearch.backend.model.dto.RerankRequest;
import com.hebproductsearch.backend.model.dto.RerankResponse;
import com.hebproductsearch.backend.model.entity.Product;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class RerankingService {
    @Value("${reranking.url}")
    private String rerankingUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public ArrayList<Product> rerank(ArrayList<Product> topN, String query, int n){
         // Build candidates payload:
        // Each element is { "product": productId, "text": product.getText() }
        List<Map<String, String>> candidates = topN.stream()
            .map(prod -> Map.of(
                "product", prod.getProductId(),
                "text", prod.getText()     // ✅ uses the Product.getText() function we added
            ))
            .collect(Collectors.toList());

        Map<String, Object> requestBody = Map.of(
            "query", query,
            "candidates", candidates
        );

        // Send POST request to Python FastAPI reranker
        Map<String, Object> response = restTemplate.postForObject(
            rerankingUrl,
            requestBody,
            Map.class
        );

        if (response == null || !response.containsKey("results")) {
            return topN; // fallback (no reranking done)
        }

        // Python returns: { "results": [ { "id": "...", "score": ... }, ... ] }
        List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");

        // Extract product_id ranking order
        List<String> rankedIds = results.stream()
            .map(entry -> (String) entry.get("id"))
            .collect(Collectors.toList());

        // Reorder original products to match ranked order
        ArrayList<Product> reranked = new ArrayList<>();
        for (String id : rankedIds) {
            for (Product prod : topN) {
                if (prod.getProductId().equals(id)) {
                    reranked.add(prod);
                    break;
                }
            }
        }

        // Trim to top N requested (e.g., 30)
        return new ArrayList<>(reranked.subList(0, Math.min(n, reranked.size())));
    }
}

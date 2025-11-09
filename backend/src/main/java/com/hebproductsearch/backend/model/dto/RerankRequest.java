package com.hebproductsearch.backend.model.dto;

import java.util.List;
import com.hebproductsearch.backend.model.entity.Product;

public class RerankRequest {
    private String query;
    private List<Product> candidates;

    public RerankRequest(String query, List<Product> candidates) {
        this.query = query;
        this.candidates = candidates;
    }

    public String getQuery() { return query; }
    public List<Product> getCandidates() { return candidates; }
}

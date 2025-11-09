package com.hebproductsearch.backend.model.dto;

import java.util.List;
import com.hebproductsearch.backend.model.entity.Product;

public class RerankResponse {
    private List<Product> reranked;

    public List<Product> getReranked() {
        return reranked;
    }
}

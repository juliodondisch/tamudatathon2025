package com.hebproductsearch.backend.service;

import java.lang.reflect.Array;
import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;

import com.hebproductsearch.backend.model.dto.DenseEmbedding;
import com.hebproductsearch.backend.model.dto.SparseEmbedding;
import com.hebproductsearch.backend.model.entity.Product;
import com.hebproductsearch.backend.repository.PostgresRepository;
import com.hebproductsearch.backend.service.Embeddings.DenseEmbeddingService;
import com.hebproductsearch.backend.service.Embeddings.SparseEmbeddingService;

public class QueryCompletionService {

    @Autowired
    private PostgresRepository postgresRepository;
    @Autowired
    private DenseEmbeddingService denseEmbeddingService;
    @Autowired
    private SparseEmbeddingService sparseEmbeddingService;

    // Goal: Construct query flow (query -> get embeddings -> perform search -> get top 10 results)
    public ArrayList<String> getTop10Responses(String query, String tableName){
        DenseEmbedding dense = denseEmbeddingService.getEmbedding(query);
        SparseEmbedding sparse = sparseEmbeddingService.getEmbedding(query);
        
        ArrayList<Product> top10Products = postgresRepository.hybridSearch(tableName, dense, sparse);
        
        ArrayList<String> top10ProductIDs = new ArrayList<String>();
        for (Product product : top10Products){
            top10ProductIDs.add(product.getID());
        }

        // Will return an ArrayList with the top 10 product ids
        return top10ProductIDs;
    }
    
}

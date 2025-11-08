package com.hebproductsearch.backend.service;

import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.hebproductsearch.backend.model.dto.DenseEmbedding;
import com.hebproductsearch.backend.model.dto.SparseEmbedding;
import com.hebproductsearch.backend.model.entity.Product;
import com.hebproductsearch.backend.repository.PostgresRepository;
import com.hebproductsearch.backend.service.Embeddings.DenseEmbeddingService;
import com.hebproductsearch.backend.service.Embeddings.SparseEmbeddingService;

@Service
public class DatabaseBuildingService {
    @Autowired
    private PostgresRepository postgresRepository;
    @Autowired
    private DenseEmbeddingService denseEmbeddingService;
    @Autowired
    private SparseEmbeddingService sparseEmbeddingService;

    // Goal: database creation flow (data -> create postgres table -> loop through data, for each: -> get embeddings -> insert data to table)
    public Boolean createDB(String tableName, ArrayList<Product> data){
        
        postgresRepository.createTable(tableName);

        for (Product product : data) {
            DenseEmbedding dense = denseEmbeddingService.getEmbedding(product.info());
            SparseEmbedding sparse = sparseEmbeddingService.getEmbedding(product.info());
            product.updateEmbeddings(dense, sparse);

            postgresRepository.insertProduct(tableName, product);
        }
        
        return postgresRepository.tableHealthCheck(tableName);
    }
}

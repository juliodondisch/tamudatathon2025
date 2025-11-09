package com.hebproductsearch.backend.service;

import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.hebproductsearch.backend.model.entity.Product;
import com.hebproductsearch.backend.repository.PostgresRepository;
import com.hebproductsearch.backend.service.Embeddings.DenseEmbeddingService;
import com.hebproductsearch.backend.service.Embeddings.ImageEmbeddingService;
import com.hebproductsearch.backend.service.Embeddings.SparseEmbeddingService;

import lombok.extern.slf4j.Slf4j;

@Service
@Slf4j
public class DatabaseBuildingService {
    @Autowired
    private PostgresRepository postgresRepository;
    @Autowired
    private DenseEmbeddingService denseEmbeddingService;
    @Autowired
    private SparseEmbeddingService sparseEmbeddingService;
    @Autowired
    private ImageEmbeddingService imageEmbeddingService;

    // Goal: database creation flow (data -> create postgres table -> loop through data, for each: -> get embeddings -> insert data to table)
    public Boolean createDB(String tableName, ArrayList<Product> data){
        log.info("Creating table: {}", tableName);
        postgresRepository.createTable(tableName);
        if( postgresRepository.tableHealthCheck(tableName)){
            log.info("created table");
        }

        log.info("Inserting data");
        for (Product product : data) {
            ArrayList<Float> dense = denseEmbeddingService.getEmbedding(product.getText());
            ArrayList<Float> sparse = sparseEmbeddingService.getEmbedding(product.getText());
            ArrayList<Float> image = imageEmbeddingService.getEmbedding(product.getProductId());
            product.updateEmbeddings(dense, sparse, image);

            log.info(product.toString());
            postgresRepository.insertProduct(tableName, product);
        }
        log.info("Data inserted");
        
        return postgresRepository.tableHealthCheck(tableName);
    }
}

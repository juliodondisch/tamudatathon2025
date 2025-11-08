package com.hebproductsearch.backend.repository;

import java.util.ArrayList;

import org.springframework.aot.hint.annotation.Reflective;
import org.springframework.stereotype.Repository;

import com.hebproductsearch.backend.model.dto.DenseEmbedding;
import com.hebproductsearch.backend.model.dto.SparseEmbedding;
import com.hebproductsearch.backend.model.entity.Product;

@Repository
public class PostgresRepository {
    
    public void createTable(String tableName){
        // Will create database table
    }

    public void insertProduct(String tableName, Object product){
        // Will insert product to database table
    }

    public ArrayList<Product> hybridSearch(String tableName, DenseEmbedding dense, SparseEmbedding sparse){
        // Will return top 10 products from search
        return new ArrayList<Product>();
    }

    public Boolean tableHealthCheck(String dbName){
        return true;
    }
}

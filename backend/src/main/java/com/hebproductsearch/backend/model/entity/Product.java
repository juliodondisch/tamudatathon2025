package com.hebproductsearch.backend.model.entity;

import org.springframework.beans.factory.annotation.Autowired;

import com.hebproductsearch.backend.model.dto.DenseEmbedding;
import com.hebproductsearch.backend.model.dto.SparseEmbedding;
import com.hebproductsearch.backend.service.Embeddings.SparseEmbeddingService;

public class Product {
    DenseEmbedding denseEmbedding;
    SparseEmbedding sparseEmbedding;
    String Name;
    String Description;

    public void updateEmbeddings(DenseEmbedding dense, SparseEmbedding sparse){
        this.denseEmbedding = dense;
        this.sparseEmbedding = sparse;
    }

    public String info(){
        return Name + Description;
    }
}

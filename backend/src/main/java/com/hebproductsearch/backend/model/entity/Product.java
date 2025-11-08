package com.hebproductsearch.backend.model.entity;

import java.util.StringTokenizer;

import org.springframework.beans.factory.annotation.Autowired;

import com.hebproductsearch.backend.model.dto.DenseEmbedding;
import com.hebproductsearch.backend.model.dto.SparseEmbedding;
import com.hebproductsearch.backend.service.Embeddings.SparseEmbeddingService;

import lombok.Data;

@Data
public class Product {
    private DenseEmbedding denseEmbedding;
    private SparseEmbedding sparseEmbedding;
    private String Name;
    private String Description;
    private String ID;

    public void updateEmbeddings(DenseEmbedding dense, SparseEmbedding sparse){
        this.denseEmbedding = dense;
        this.sparseEmbedding = sparse;
    }

    public String info(){
        return Name + Description;
    }
}

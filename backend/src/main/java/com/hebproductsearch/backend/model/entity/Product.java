package com.hebproductsearch.backend.model.entity;

import java.util.ArrayList;
import java.util.StringTokenizer;

import org.springframework.beans.factory.annotation.Autowired;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;

@Data
public class Product {
    @JsonProperty("product_id") 
    private String productId;
    private ArrayList<Float> denseEmbedding;
    private ArrayList<Float> sparseEmbedding;
    private String title;
    private String description;
    private String brand;
    @JsonProperty("category_path") 
    private String categoryPath;
    @JsonProperty("safety_warning") 
    private String safetyWarning;
    private String ingredients;

    public void updateEmbeddings(ArrayList<Float> dense, ArrayList<Float> sparse){
        this.denseEmbedding = dense;
        this.sparseEmbedding = sparse;
    }

    public String info(){
        return title + description;
    }
    
}

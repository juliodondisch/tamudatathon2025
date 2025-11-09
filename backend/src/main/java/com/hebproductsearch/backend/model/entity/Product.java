package com.hebproductsearch.backend.model.entity;

import java.util.ArrayList;
import java.util.StringTokenizer;

import org.springframework.beans.factory.annotation.Autowired;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Product {
    @JsonProperty("product_id") 
    private String productId;
    private ArrayList<Float> denseEmbedding;
    private ArrayList<Float> sparseEmbedding;
    private ArrayList<Float> imageEmbedding;
    private String title;
    private String description;
    private String brand;
    @JsonProperty("category_path") 
    private String categoryPath;
    @JsonProperty("safety_warning") 
    private String safetyWarning;
    private String ingredients;

    public void updateEmbeddings(ArrayList<Float> dense, ArrayList<Float> sparse, ArrayList<Float> image){
        this.denseEmbedding = dense;
        this.sparseEmbedding = sparse;
        this.imageEmbedding = image;
    }

    public String info(){
        return title + description;
    }

    public String getText() {
        StringBuilder sb = new StringBuilder();

        if (title != null && !title.isBlank()) {
            sb.append("Title: ").append(title).append(". ");
        }

        if (description != null && !description.isBlank()) {
            sb.append("Description: ").append(description).append(". ");
        }

        if (brand != null && !brand.isBlank()) {
            sb.append("Brand: ").append(brand).append(". ");
        }

        if (categoryPath != null && !categoryPath.isBlank()) {
            sb.append("Category: ").append(categoryPath).append(". ");
        }

        if (safetyWarning != null && !safetyWarning.isBlank()) {
            sb.append("Safety Warning: ").append(safetyWarning).append(". ");
        }

        if (ingredients != null && !ingredients.isBlank()) {
            sb.append("Ingredients: ").append(ingredients).append(". ");
        }

        return sb.toString().trim();
    }

    
}

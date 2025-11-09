package com.hebproductsearch.backend.repository;

import java.util.ArrayList;

import org.springframework.aot.hint.annotation.Reflective;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.hebproductsearch.backend.model.entity.Product;

@Repository
public class PostgresRepository{

    private final JdbcTemplate jdbc;
    private final ObjectMapper mapper = new ObjectMapper();

    public PostgresRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }
    
    public void createTable(String tableName){
        // Will create database table
        String sql = """
            CREATE TABLE IF NOT EXISTS %s (
                product_id TEXT PRIMARY KEY,
                dense_embedding VECTOR(384),
                sparse_embedding VECTOR(1000),
                title TEXT,
                description TEXT,
                brand TEXT,
                category_path TEXT,
                safety_warning TEXT,
                ingredients TEXT
            )
        """.formatted(tableName);

        jdbc.execute(sql);
    }

    public void insertProduct(String tableName, Product product) {
        try {
            // Convert ArrayList<Float> to PostgreSQL vector format '[x1,x2,...,xn]'
            String denseVectorStr = product.getDenseEmbedding().toString()
                .replace("[", "'[")
                .replace("]", "]'");
            String sparseVectorStr = product.getSparseEmbedding().toString()
                .replace("[", "'[")
                .replace("]", "]'");

            String sql = String.format("""
                INSERT INTO %s (
                    product_id, dense_embedding, sparse_embedding,
                    title, description, brand, category_path,
                    safety_warning, ingredients
                ) VALUES (?, %s, %s, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO NOTHING
                """, tableName, denseVectorStr, sparseVectorStr);

            jdbc.update(sql,
                product.getProductId(),
                product.getTitle(),
                product.getDescription(), 
                product.getBrand(),
                product.getCategoryPath(),
                product.getSafetyWarning(),
                product.getIngredients()
            );
        } catch(Exception e) {
            throw new RuntimeException("Failed to insert: " + e.getMessage(), e);
        }
    }

    public ArrayList<Product> hybridSearch(String tableName, ArrayList<Float> dense, ArrayList<Float> sparse){
        // Will return top 10 products from search
        String denseVectorStr = dense.toString()
                .replace("[", "[")
                .replace("]", "]");
        String sparseVectorStr = sparse.toString()
                .replace("[", "[")
                .replace("]", "]");

        String sql = """
            SELECT product_id, title, description, brand, 
                   category_path, safety_warning, ingredients,
                   dense_embedding <#> '%s'::vector AS similarity
            FROM %s
            ORDER BY similarity
            LIMIT 10;
        """.formatted(denseVectorStr, tableName);
        
        return new ArrayList<>(jdbc.query(sql, (rs, rowNum) -> 
            new Product(
                rs.getString("product_id"),
                dense,
                sparse,
                rs.getString("title"),
                rs.getString("description"),
                rs.getString("brand"),
                rs.getString("category_path"),
                rs.getString("safety_warning"),
                rs.getString("ingredients")
            )
        ));
    }

    public Boolean tableHealthCheck(String tableName){
        try {
            jdbc.queryForObject("SELECT 1 FROM " + tableName + " LIMIT 1", Integer.class);
            return true;
        } catch(Exception e){
            return false;
        }
    }
}

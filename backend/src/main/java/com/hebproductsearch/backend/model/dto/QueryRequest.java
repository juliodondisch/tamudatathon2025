package com.hebproductsearch.backend.model.dto;

import lombok.Data;

@Data
public class QueryRequest {
    String query;
    String tableName;
}

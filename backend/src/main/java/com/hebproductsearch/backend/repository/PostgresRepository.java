package com.hebproductsearch.backend.repository;

import org.springframework.aot.hint.annotation.Reflective;
import org.springframework.stereotype.Repository;

@Repository
public class PostgresRepository {
    
    public void createTable(String dbName){

    }

    public void insertProduct(Object product){

    }

    public Boolean tableHealthCheck(String dbName){
        return true;
    }
}

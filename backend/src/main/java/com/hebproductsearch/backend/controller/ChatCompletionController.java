package com.hebproductsearch.backend.controller;

import com.hebproductsearch.backend.model.dto.Product;
import com.hebproductsearch.backend.service.DatabaseBuildingService;
import com.hebproductsearch.backend.service.QueryCompletionService;

import java.sql.DatabaseMetaData;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("")
public class ChatCompletionController {

    private static final Logger log = LoggerFactory.getLogger(ChatCompletionController.class);

    @Autowired
    private QueryCompletionService queryCompletion;

    @Autowired
    private DatabaseBuildingService databaseBuilder;

    @PostMapping("/create-db") 
    public ResponseEntity<Boolean> chatCompletions(@PathVariable String dbName, @RequestBody ArrayList<Product> data) {
        try {
            Boolean success = databaseBuilder.createDB(dbName, data);
            if (success){
                return ResponseEntity.status(200).body(true);
            }
            else{
                return ResponseEntity.status(400).body(false);
            }
        } 
        catch (Exception e) {
            log.error("Database Creation Failed", e);
            return ResponseEntity.status(500).body(false);
        }
    }
}

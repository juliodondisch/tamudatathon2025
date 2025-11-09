package com.hebproductsearch.backend.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.hebproductsearch.backend.model.entity.Product;
import com.hebproductsearch.backend.repository.PostgresRepository;

@Service
public class HybridSearchService {
    @Autowired
    private PostgresRepository postgresRepository;

    public ArrayList<Product> hybridSearch(String tableName, ArrayList<Float> dense, ArrayList<Float> sparse, ArrayList<Float> image){
        
        ArrayList<Product> denseResults = postgresRepository.denseSearch(tableName, dense, sparse, image);
        ArrayList<Product> sparseResults = postgresRepository.sparseSearch(tableName, dense, sparse, image);
        ArrayList<Product> imageResults = postgresRepository.imageSearch(tableName, dense, sparse, image);

        double a = 0.6;
        double b = 0.2;
        double c = 0.2;
        int k = 60;

        Map<String, Integer> denseRank = new HashMap<>();
        for (int i = 0; i < denseResults.size(); i++) {
            denseRank.put(denseResults.get(i).getProductId(), i + 1);
        }
        
        Map<String, Integer> sparseRank = new HashMap<>();
        for (int i = 0; i < sparseResults.size(); i++) {
            sparseRank.put(sparseResults.get(i).getProductId(), i + 1);
        }

        Map<String, Integer> imageRank = new HashMap<>();
        for (int i = 0; i < imageResults.size(); i++) {
            imageRank.put(imageResults.get(i).getProductId(), i + 1);
        }
        
        Map<String, Double> scores = new HashMap<>();
        
        for (Product p : denseResults) {
            scores.put(p.getProductId(), 0.0);
        }
        for (Product p : sparseResults) {
            scores.put(p.getProductId(), 0.0);
        }
        for (Product p : imageResults) {
            scores.put(p.getProductId(), 0.0);
        }
        
        for (String id : scores.keySet()) {
            double denseScore = denseRank.containsKey(id) ? a * (1.0 / (k + denseRank.get(id))) : 0.0;
            double sparseScore = sparseRank.containsKey(id) ? b * (1.0 / (k + sparseRank.get(id))) : 0.0;
            double imageScore = imageRank.containsKey(id) ? c * (1.0 / (k + imageRank.get(id))) : 0.0;
            scores.put(id, denseScore + sparseScore + imageScore);
        }
        
        ArrayList<Product> fused = new ArrayList<>();
        
        Map<String, Product> lookup = new HashMap<>();
        denseResults.forEach(p -> lookup.put(p.getProductId(), p));
        sparseResults.forEach(p -> lookup.putIfAbsent(p.getProductId(), p));
        imageResults.forEach(p -> lookup.putIfAbsent(p.getProductId(), p));
        fused.addAll(lookup.values());
        fused.sort((p1, p2) -> Double.compare(scores.get(p2.getProductId()), scores.get(p1.getProductId())));
        // cutting to 64 max because that's optimal for bge reranker
        return new ArrayList<>(fused.subList(0, Math.min(64, fused.size())));

    }
}

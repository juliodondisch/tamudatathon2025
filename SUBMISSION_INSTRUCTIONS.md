# HEB Datathon Submission Instructions - Team Penguin

## Overview

This document contains step-by-step instructions for submitting your predictions to the HEB Datathon official evaluation system.

## Submission Files

- **submission.json**: Your predictions in the required format (9,550 entries)
- **predictions.json**: Original predictions with relevance scores (backup)
- **Team Name**: penguin

## Submission Details

- Total queries: 191 test queries
- Products per query: 50 ranked products
- Total entries: 9,550
- Format: Validated and ready to submit
- Model used: Fine-tuned sentence transformer (+189% improvement over baseline)

## Step-by-Step Submission Process

### Step 1: Navigate to HEB Repository

```bash
cd /home/deep/Datathon/tamu-2025-heb-main/tamu-2025-heb-main/
```

### Step 2: Update from Main Branch

```bash
git checkout main
git pull origin main
```

### Step 3: Create Submission Branch

```bash
git checkout -b team_penguin/submission-run-01
```

Note: Increment the run number (01, 02, 03) for each resubmission.

### Step 4: Copy Submission File to Team Folder

```bash
cp /home/deep/Datathon/tamudatathon2025/submission.json teams/team_penguin/submission.json
```

### Step 5: Commit Changes

```bash
git add teams/team_penguin/submission.json
git commit -m "team_penguin | submission run 01 | fine-tuned sentence transformer"
```

### Step 6: Push to Remote

```bash
git push origin team_penguin/submission-run-01
```

### Step 7: Create Merge Request on GitLab

1. Go to your GitLab repository web interface
2. Click "Create Merge Request"
3. Configure the merge request:
   - Source branch: team_penguin/submission-run-01
   - Target branch: main
   - MR Title: team_penguin | submission #01 | fine-tuned sentence transformer
   - Description (optional): Fine-tuned model with 189% improvement in Spearman correlation
4. Click "Create Merge Request"

### Step 8: Monitor CI Pipeline

After creating the MR, the GitLab CI pipeline will automatically run:

1. **Stage 1: unit_tests** - Verifies package integrity
2. **Stage 2: validate-submission** - Checks format and coverage
3. **Stage 3: score-submission** - Computes metrics
4. **Stage 4: persist_score** - Saves scores and metadata
5. **Stage 5: build_leaderboard** - Updates leaderboard

### Step 9: Review Results

Check the MR page for pipeline artifacts:
- validation_report.json - Format validation results
- score_report.json - Your performance metrics
- metadata.json - Submission metadata

## Evaluation Metrics

Your submission will be scored on:

- **nDCG@10**: Ranking quality with graded relevance (0-3 scale)
- **MAP@10**: Mean Average Precision at 10 results
- **P@5**: Precision at top 5 results
- **R@10**: Recall at 10 results

Final Score Formula:
```
Composite(q) = 0.30 * nDCG@10(q) + 0.30 * AP@20(q) + 0.25 * R@30(q) + 0.15 * P@10(q)
Weighted Final = 0.0 * composite_real + 1.0 * composite_synthetic
```

Since only synthetic test queries are provided, your score will be 100% based on synthetic query performance.

## Validation Requirements

Your submission must meet these requirements (already validated):

- Must be valid JSON array
- At least 10 results per query
- Ranks strictly sequential (1, 2, 3, ...)
- Each (query_id, product_id) pair is unique
- All test queries are covered
- All product_ids exist in the catalog

Status: PASSED

## Submission Limits

- Maximum 3 submissions per day
- Submissions failing validation do not count toward limit
- Latest successful score replaces previous scores on leaderboard

## Resubmission Process

If you need to resubmit:

1. Return to main branch:
   ```bash
   cd /home/deep/Datathon/tamu-2025-heb-main/tamu-2025-heb-main/
   git checkout main
   git pull origin main
   ```

2. Create new branch with incremented run number:
   ```bash
   git checkout -b team_penguin/submission-run-02
   ```

3. Copy updated submission:
   ```bash
   cp /home/deep/Datathon/tamudatathon2025/submission.json teams/team_penguin/submission.json
   ```

4. Commit and push:
   ```bash
   git add teams/team_penguin/submission.json
   git commit -m "team_penguin | submission run 02 | description of changes"
   git push origin team_penguin/submission-run-02
   ```

5. Create new Merge Request with updated title

## Quick Reference Commands

All commands in sequence (copy-paste ready):

```bash
# Navigate to HEB repo
cd /home/deep/Datathon/tamu-2025-heb-main/tamu-2025-heb-main/

# Update from main
git checkout main
git pull origin main

# Create submission branch
git checkout -b team_penguin/submission-run-01

# Copy submission file
cp /home/deep/Datathon/tamudatathon2025/submission.json teams/team_penguin/submission.json

# Commit and push
git add teams/team_penguin/submission.json
git commit -m "team_penguin | submission run 01 | fine-tuned sentence transformer"
git push origin team_penguin/submission-run-01
```

Then create the Merge Request on GitLab web interface.

## Troubleshooting

### Validation Fails

If validation fails, check:
- All query IDs from queries_synth_test.json are present
- Ranks are sequential (1, 2, 3, ...) for each query
- No duplicate (query_id, product_id) pairs
- All product IDs exist in products.json

### Pipeline Fails

If the CI pipeline fails:
- Check the pipeline logs in GitLab
- Review validation_report.json artifact
- Ensure submission.json is in correct location
- Verify JSON is properly formatted

### Team Folder Issues

If teams/team_penguin/ doesn't exist:
- Create it: `mkdir -p teams/team_penguin`
- Then copy submission file

## Model Information

This submission uses a fine-tuned sentence transformer:

- Base model: all-MiniLM-L6-v2
- Fine-tuning: 3 epochs on query-product pairs
- Embedding dimension: 384
- Performance: +189% improvement in Spearman correlation vs baseline
- Training data: 10,974 query-product pairs with relevance labels (0-3)

### Relevance Score Thresholds

Products are ranked by similarity score with these relevance mappings:
- Similarity >= 0.7 → Relevance 3 (highly relevant)
- Similarity 0.5-0.7 → Relevance 2 (moderately relevant)
- Similarity 0.3-0.5 → Relevance 1 (slightly relevant)
- Similarity < 0.3 → Relevance 0 (not relevant)

## Support

For questions about:
- Submission format: See README.md in HEB repository
- Model details: See SENTENCE_TRANSFORMER_README.md
- Integration: See INTEGRATION_GUIDE.md
- Testing: See TESTING_GUIDE.md

## Files Reference

In this repository (tamudatathon2025):
- submission.json - Ready to submit
- predictions.json - Original predictions with relevance scores
- generate_predictions.py - Script to regenerate predictions
- convert_to_submission_format.py - Script to convert format
- model_interface_v2.py - Model interface
- output/heb-semantic-search/ - Fine-tuned model weights

In HEB repository (tamu-2025-heb-main/tamu-2025-heb-main):
- teams/team_penguin/submission.json - Submit here
- data/queries_synth_test.json - Test queries
- data/products.json - Product catalog
- README.md - Official documentation

# Project 1: SMS Spam Detection System

## Goal
Create a machine learning system that reads a text message and
automatically determines whether it is spam or legitimate (ham).

## Dataset
- **Name:** SMS Spam Collection
- **Origin:** UCI Repository / Kaggle
- **Total Messages:** 5,572
- **Legitimate (ham):** 4,825 (86.6%)
- **Spam:** 747 (13.4%)
- **Format:** CSV with category and text columns

## Methodology

### Phase 1: Data Loading
- Imported CSV using pandas
- Handled encoding and cleaned column structure

### Phase 2: Data Exploration
- Examined spam vs ham balance
- Measured character and word counts per category
- Discovered spam messages tend to be longer
- Built bar charts, pie charts, and distribution histograms

### Phase 3: Text Cleaning Pipeline
1. Case normalization to lowercase
2. Removed numbers, symbols, and punctuation
3. Filtered out English stopwords
4. Applied Porter Stemming to get word roots
5. Discarded very short words (2 chars or less)

##Example:
Input: "CONGRATULATIONS!! You've WON $5000! Call NOW!!!"
Output: "congratul won call"


### Phase 4: Feature Engineering
- Used TF-IDF to convert text into numerical vectors
- Words important in specific messages get higher scores
- Selected top 3000 most discriminating words

### Phase 5: Data Splitting
- 80% training, 20% testing
- Stratified to preserve class ratios

### Phase 6: Model Development

**Naive Bayes:** Probabilistic model ideal for text tasks.
Calculates spam probability based on word frequencies. Fast and reliable.

**Logistic Regression:** Linear classifier that learns a decision
boundary. Serves as comparison against Naive Bayes.

### Phase 7: Performance Assessment
- Accuracy, Precision, Recall, F1 Score
- Confusion matrices for detailed error breakdown
- Side-by-side model comparison

### Phase 8: Practical Validation
Tested winning model with hand-crafted messages to confirm
real-world applicability on unseen text inputs.

## Performance

| Classifier          | Accuracy | Precision | Recall |
|---------------------|----------|-----------|--------|
| Naive Bayes         | ~97%     | ~100%     | ~80%   |
| Logistic Regression | ~96%     | ~98%      | ~78%   |

**Selected Model:** Naive Bayes

## Output Files
| File | Content |
|------|---------|
| exploration_charts.png | EDA visualizations |
| evaluation_matrices.png | Confusion matrices |
| accuracy_comparison.png | Model accuracy comparison |

## How to Run
```bash
pip install -r requirements.txt
python spam_classifier.py

##Takeaways
1. Raw text needs extensive cleaning before ML can use it
2. TF-IDF captures word importance effectively
3. Naive Bayes is excellent for text classification
4. Comparing models is essential for best selection
5. Multiple metrics beat relying on accuracy alone
##Libraries
Python, pandas, numpy, scikit-learn, NLTK, matplotlib, seaborn 

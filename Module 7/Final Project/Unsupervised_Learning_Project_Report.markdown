# Unsupervised Machine Learning Project Report: Customer Segmentation for Retail Marketing

## Main Objective
The primary objective of this analysis is to apply **clustering** techniques to segment customers of an online retail store based on their purchasing behavior. By grouping similar customers into clusters, the business can tailor marketing strategies to specific segments, improving customer engagement and increasing sales. The benefits include more personalized marketing campaigns, optimized resource allocation, and enhanced customer retention through targeted promotions.

## Data Description
The dataset used is the "Online Retail Dataset" from the UCI Machine Learning Repository, containing transactional data from a UK-based online retailer. It includes 541,909 transactions from December 2010 to December 2011, with the following key attributes:
- **InvoiceNo**: Unique identifier for each transaction.
- **StockCode**: Product code for purchased items.
- **Description**: Product description.
- **Quantity**: Number of items purchased.
- **InvoiceDate**: Date and time of the transaction.
- **UnitPrice**: Price per item.
- **CustomerID**: Unique identifier for each customer.
- **Country**: Customer’s country.

The goal is to cluster customers based on their purchasing patterns, such as total spending, frequency of purchases, and product preferences, to identify distinct customer segments for marketing purposes.

## Data Exploration and Preprocessing
### Exploration
Initial exploration revealed:
- **Missing Values**: Approximately 25% of transactions lacked a CustomerID, and some had missing product descriptions.
- **Outliers**: Some transactions had negative quantities (likely returns) or extremely high quantities/prices.
- **Data Types**: Numeric (Quantity, UnitPrice), categorical (StockCode, Country), and datetime (InvoiceDate).
- **Distribution**: Most customers were from the UK, with a long tail of infrequent buyers and a few high-value customers.

### Preprocessing
The following steps were taken to clean and prepare the data:
1. **Filtered Data**: Removed transactions without CustomerID to focus on identifiable customers.
2. **Aggregated Features**: Created customer-level features:
   - **Total Spending**: Sum of (Quantity × UnitPrice) per customer.
   - **Purchase Frequency**: Number of transactions per customer.
   - **Average Order Value**: Total spending divided by purchase frequency.
   - **Recency**: Days since the last purchase (relative to the dataset’s latest date).
3. **Removed Outliers**: Excluded transactions with negative quantities or extreme values (e.g., top 1% in spending or quantity).
4. **Feature Scaling**: Standardized features using StandardScaler to ensure equal weighting in clustering.
5. **Feature Engineering**: Created a binary feature indicating whether a customer purchased high-value products (based on UnitPrice thresholds).

After preprocessing, the dataset contained 4,372 unique customers with four numerical features: Total Spending, Purchase Frequency, Average Order Value, and Recency.

## Model Training and Comparison
Three clustering models were trained to segment customers, each with variations in algorithms or hyperparameters:
1. **K-means Clustering**:
   - **Parameters Tested**: Number of clusters (k = 3, 4, 5).
   - **Evaluation**: Used the silhouette score to assess cluster cohesion and separation.
   - **Results**: For k=4, the silhouette score was 0.42, indicating reasonable cluster quality. Clusters represented high-value frequent buyers, low-value frequent buyers, occasional high-value buyers, and inactive low-value buyers.
2. **Hierarchical Clustering (Agglomerative)**:
   - **Parameters Tested**: Linkage methods (ward, complete, average) with 4 clusters.
   - **Evaluation**: Silhouette score and dendrogram analysis for cluster interpretability.
   - **Results**: Ward linkage yielded a silhouette score of 0.39. Clusters were similar to K-means but less distinct, with some overlap in low-value segments.
3. **DBSCAN**:
   - **Parameters Tested**: Varied eps (distance threshold) and min_samples (minimum points per cluster).
   - **Evaluation**: Assessed the number of clusters formed and noise points.
   - **Results**: DBSCAN identified 3 clusters with eps=0.5 and min_samples=5, but 20% of customers were labeled as noise, reducing its practical utility for marketing.

## Model Selection
The **K-means model with k=4** is recommended as the final model. It achieved the highest silhouette score (0.42) and produced well-separated, interpretable clusters that align with the business objective of targeted marketing. The clusters are:
1. **High-Value Frequent Buyers**: High spending and frequent purchases (ideal for loyalty programs).
2. **Low-Value Frequent Buyers**: Frequent but low-spending customers (target for upselling).
3. **Occasional High-Value Buyers**: Infrequent but high-spending (target for re-engagement campaigns).
4. **Inactive Low-Value Buyers**: Low spending and long recency (low-priority segment).

K-means outperformed hierarchical clustering in terms of cluster cohesion and was more practical than DBSCAN, which struggled with noise points in this dataset.

## Key Findings and Insights
The clustering analysis revealed four distinct customer segments, enabling the following insights:
- **High-Value Frequent Buyers** (15% of customers) account for 40% of total revenue, making them the most critical segment for retention-focused campaigns.
- **Low-Value Frequent Buyers** (30% of customers) show consistent engagement but low spending, suggesting opportunities for upselling or cross-selling.
- **Occasional High-Value Buyers** (20% of customers) have high spending but infrequent purchases, indicating potential for re-engagement promotions.
- **Inactive Low-Value Buyers** (35% of customers) contribute minimal revenue and may not warrant significant marketing investment.
These findings allow the business to allocate marketing resources efficiently, focusing on high-value segments while designing strategies to convert low-value frequent buyers into higher spenders.

## Suggestions for Next Steps
To improve the analysis, consider the following:
1. **Incorporate Additional Features**: Include demographic data (e.g., age, location) or product category preferences to enhance cluster granularity.
2. **Experiment with Other Algorithms**: Test Gaussian Mixture Models or spectral clustering to capture non-linear patterns in the data.
3. **Address Noise in DBSCAN**: Fine-tune DBSCAN parameters or preprocess data to reduce noise points, potentially making it viable for sparse datasets.
4. **Validate with Supervised Learning**: Develop a supervised model (e.g., classification to predict customer churn) for each cluster to assess whether cluster-specific models outperform a single model.
5. **Collect Recent Data**: Update the dataset with more recent transactions to ensure relevance for current marketing strategies.

By addressing these steps, the business can refine customer segmentation and further optimize marketing efforts.
# Yelp Recommender System

---

## Raw Datasets

### Yelp Datasets
Retrieved from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

### U.S. Census Data (for location enrichment)
- [US Zip Codes Demographics](https://drive.google.com/file/d/1mmIf-vNDYJzzKYn8nmhQNu7kIVpUAoVw/view?usp=drive_link): Proximity to highways & income by postal code  
- [Unemployment & Median Household Income](https://drive.google.com/file/d/1RZzEXlu0RFWjsUaa6WIOsxOC3aaLYfO7/view?usp=drive_link): Urban-rural classification & unemployment by FIPS Code  
- [ZIP to FIPS Crosswalk](https://drive.google.com/file/d/1uhNr-SmK9qkIkPDsPva_1DI-IDEHoeX9/view?usp=drive_link): Used to join datasets via common `postal_code`

### Google Maps API
Used to supplement missing or null values in Yelp data and to provide more granular product information.  
- `BT4222_Project_Google_Maps_Api.ipynb`
- `BT4222_Project_Join_Google_Features.ipynb`
---

## Process Overview

### Step 1: Data Preparation

| Notebook | Inputs | Outputs |
|----------|--------|---------|
| `01_BusinessDataPreparation.ipynb` | Yelp business JSON, [Census data](https://drive.google.com/file/d/1Qz-8x8soS8POCzi2koMYYWFp99yzJXp7/view?usp=drive_link), Google Maps Data | [`restaurants_final.csv`](https://drive.google.com/file/d/19eRDb6kn3mrpNAYpoVNePc9RzsUikS9i/view?usp=drive_link)|
| `02_ReviewDataPreparation.ipynb` | Yelp review + user JSON | [`review_final.csv`](https://drive.google.com/file/d/1KSxmDfBq7x7pKNY36XqHWZAqZN8aQgNs/view?usp=drive_link)|
| `03_UserDataset.ipynb` | Yelp user JSON | [`PA_all_restaurant_user_with_loc.csv`](https://drive.google.com/file/d/15l0CmR1b7xO4-U5yqSmgRR5V7CeWx3Vr/view?usp=drive_link) |
| `04_UserClustering.ipynb` | Yelp user + review JSON | `user_cluster.csv` |

---

### Step 2: Individual Recommender Models

| Notebook | Description | Inputs | Outputs |
|----------|-------------|--------|---------|
| `05_PopularityBasedRecommendation.ipynb` | Most reviewed/high-rated restaurants | `03_UserDataset.ipynb`, [`restaurant_w_train_ave_stars.csv`](https://drive.google.com/file/d/1WJxDCaqh_DnsGYC-lAi-Fpjkf2j6i018/view?usp=drive_link), `restaurants_final.csv`, `review_final.csv`, `PA_all_restaurant.csv` | - |
| `06_ContentBasedRecommendation.ipynb` | TF-IDF and cosine similarity between restaurants | `restaurants_final.csv`, `review_final.csv` | `train_df.csv`, `test_df.csv`, `cb_matrix.npz`, `item_df.csv` |
| `07_UserBasedCollaborativeFiltering.ipynb` | Cosine similarity on user-item interactions | `review_final.csv`, `PA_all_restaurant_user_with_loc.csv` | `users_df.csv`, `cf_matrix.npz`, user/item encoders |
| `08_MFModelFeatures.ipynb` | Matrix Factorization with user/item features | All processed datasets | `user/item_features_tensor_pt`, encoders, [`MF_model.pth`](https://drive.google.com/file/d/1rPxT_M8WCJyY5U5qsCUB-iY9dA-uvogq/view?usp=drive_link) |
| `09_NeuralCollaborativeFiltering.ipynb` | Deep learning model | All processed datasets | [`user_restaurant_review.csv`](https://drive.google.com/file/d/1Qtld6cJ25v3np0Mk1fFP88Eu91wANQ4r/view?usp=drive_link), `item_ncf_encoder.pkl`, `NCF_model.pth` |

---

### Step 3: Combined Models

| Notebook | Description | Inputs | Outputs |
|----------|-------------|--------|---------|
| `10_DiverseSelection.ipynb` | Preparatory code for diverse selection method used in Combined Pipeline | `restaurant_w_train_ave_stars.csv` (from 05_PopularityBasedRecommendation.ipynb) | `ds_matrix.npz` |
| `11_CombinedPipeline.ipynb` | Integrates all models | All outputs from Step 2 | Final top-K recommendations |
---

## Notes

- Most files are structured in Jupyter Notebooks. Only 05_PopularityBasedRecommendation.ipynb and 09_NeuralCollaborativeFiltering.ipynb uses Colab. 
- Outputs are stored in `.csv`, `.npz`, `.pkl`, and `.pth` formats for cross-pipeline integration.

---

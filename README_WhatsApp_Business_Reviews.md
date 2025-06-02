
# 📊 WhatsApp Business Reviews: App Store

This project leverages the [WhatsApp Business Reviews: App Store](https://www.kaggle.com/datasets/kanchana1990/whatsapp-business-reviews-app-store) dataset, which comprises user reviews from the App Stores of three distinct nations. The dataset provides valuable insights into user sentiments, feedback, and experiences with the WhatsApp Business application.

## 📁 Dataset Overview

- **Source**: Kaggle Dataset by [kanchana1990](https://www.kaggle.com/kanchana1990)
- **Size**: Approximately 247 KB
- **Format**: CSV
- **Languages**: Primarily English
- **Fields**:
  - `review`: The text content of the user review
  - `rating`: Numerical rating provided by the user
  - `country`: Country from which the review was submitted
  - `date`: Date of the review
  - `version`: App version at the time of review

## 🎯 Project Objectives

- **Sentiment Analysis**: Determine the overall sentiment (positive, negative, neutral) expressed in user reviews.
- **Topic Modeling**: Identify prevalent themes and topics discussed by users.
- **Trend Analysis**: Observe how user sentiments and topics evolve over time and across different countries.
- **Feature Extraction**: Highlight frequently mentioned features, issues, or suggestions.

## 🛠️ Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/whatsapp-business-reviews.git
   cd whatsapp-business-reviews
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/kanchana1990/whatsapp-business-reviews-app-store).
   - Download the CSV file and place it in the `data/` directory of this project.

## 📊 Usage

The project includes Jupyter notebooks for data exploration, preprocessing, and analysis.

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Navigate to the Notebook**:
   - Open `notebooks/analysis.ipynb` to explore the data and perform analyses.

## 📈 Sample Visualizations

*Include sample plots or word clouds here to showcase insights derived from the data.*


---

*Note: Ensure you have the necessary permissions and adhere to the dataset's licensing terms when using and sharing this data.*

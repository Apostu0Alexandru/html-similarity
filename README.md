# HTML Document Similarity Analyzer

A sophisticated algorithm that groups HTML documents based on how users perceive them in a web browser. This multi-dimensional analyzer evaluates content, visual appearance, DOM structure, and semantic meaning to identify similar webpages with high accuracy.

---

## 🌟 Project Overview

This project tackles the challenge of grouping HTML documents based on their similarity from a user's perspective. The key insight driving this solution is that users primarily perceive web pages through their visual appearance, content, and structure—not through raw HTML code.

By analyzing multiple dimensions of similarity, this algorithm effectively identifies pages that would appear similar to users browsing the web, even when their underlying HTML implementations differ.

---

## 🎯 Methodology: A Multi-Dimensional Approach

Rather than relying on a single metric, this solution analyzes HTML documents across four key dimensions that mirror how users perceive web pages:

### 1. Content Similarity (30%)
Users immediately notice what they can read on the page. The algorithm extracts visible text and applies TF-IDF vectorization with cosine similarity to identify documents with similar textual content.

### 2. Visual Appearance (40%)
The most immediate aspect users perceive is how a page looks. The algorithm captures this through:
- **Color scheme analysis**: Extracting and comparing dominant colors
- **CSS property comparison**: Analyzing layout, fonts, and styling
- **Visual fingerprinting**: Creating simplified visual representations of page layouts

### 3. DOM Structure (20%)
The structural organization of a page affects how users interact with it. The algorithm analyzes:
- **Tag frequencies**: Counting HTML elements to identify similar page structures
- **Hierarchical DOM analysis**: Comparing the tree-like organization of elements
- **Tag importance weighting**: Giving higher importance to structural elements like headers

### 4. Semantic Meaning (10%)
Understanding the purpose of page sections enhances similarity detection. The algorithm identifies:
- **Semantic HTML5 elements**: Detecting ``, ``, ``, etc.
- **Above-the-fold content prioritization**: Focusing on what users see first
- **Navigation patterns**: Comparing menu structures and interaction points

---

## 🔍 The Algorithm in Action

The algorithm processes HTML documents in several steps:

1. **Feature Extraction**: Analyzes each HTML file to extract multi-dimensional features.
2. **Similarity Computation**: Calculates similarity scores between all document pairs.
3. **Threshold-Based Grouping**: Groups documents with similarity scores above the configured threshold.

---

## 📂 Web Scraping Functionality

To test the algorithm on real-world data, this project includes a web scraping script (`web_scraping.py`). The script:
1. Scrapes live HTML pages from a list of websites.
2. Saves them as `.html` files in a specified directory (`scraped_html`).
3. Enables you to run the similarity analyzer on real-world datasets.

### Example Usage:
```bash
python web_scraping.py  # Scrape live HTML pages
python html_similarity.py scraped_html --visualize --output results.json  # Analyze scraped data
```

---

## 📊 Example Results

For a directory containing 10 HTML files, the output might look like:

```
Group 1: ['aemails.org.html', 'afro-pari.com.html', 'aevesdk3.com.html', 'aigner-haag.at.html', 'aerex.eu.html', 'akashinime.guru.html', 'ahamconsumerconnections.org.html', 'ai-center.online.html', 'ahbynmkkmnfu.shop.html']
Group 2: ['aitoka.shop.html']
```

The visualization module can also generate similarity matrices as heatmaps.

---

## 🛠️ Paths Explored and Decisions Made

### What Worked Well:
1. **Multi-dimensional similarity metrics**: Combining different aspects of similarity proved more effective than relying on any single metric.
2. **Visual fingerprinting**: Captured layout patterns efficiently without requiring full rendering.
3. **Weighted approach**: Allowed fine-tuning based on dataset characteristics.

### Approaches Considered But Not Implemented:
1. **Full-page rendering using headless browsers:** Avoided due to computational overhead.
2. **Neural network-based embeddings:** Excluded to keep the solution lightweight and interpretable.
3. **Graph-based clustering:** Opted for simpler threshold-based grouping for better interpretability.

---

## 💻 Usage Instructions

### Installation:
```bash
# Clone the repository
git clone https://github.com/Apostu0Alexandru/html-similarity.git
cd html-similarity

# Install required dependencies
pip install -r requirements.txt
```

### Basic Usage:
```bash
python html_similarity.py /path/to/html/files --visualize --output results.json
```

### Advanced Options:
```bash
python html_similarity.py /path/to/html/files \
  --content-weight 0.3 \
  --structure-weight 0.2 \
  --visual-weight 0.4 \
  --semantic-weight 0.1 \
  --threshold 0.75 \
  --visualize \
  --output results.json
```

---

## 📊 Scalability Considerations

This solution is designed to handle large datasets efficiently:
- Optimized feature extraction focuses only on relevant HTML features.
- Vectorized operations use NumPy for efficient similarity calculations.
- Progressive processing handles directories one at a time to manage memory usage.
- Optional features like screenshot rendering can be disabled for faster processing.

For truly massive datasets (billions of records), the algorithm could be extended with:
- Locality-Sensitive Hashing (LSH) for efficient approximate nearest neighbor search.
- Distributed computing for processing documents in parallel across multiple machines.
- Incremental clustering for adding new documents without reprocessing the entire dataset.

---

## 🔮 Future Enhancements

1. Interactive visualization tools for exploring document similarities.
2. Adaptive weighting systems to automatically determine optimal weights based on dataset characteristics.
3. Temporal analysis for tracking changes in documents over time.

---

## 🏁 Conclusion

This HTML similarity analyzer effectively groups documents based on how users perceive them in a web browser by combining content, visual appearance, structure, and semantics into a unified model.

The inclusion of web scraping functionality makes it easy to test the algorithm on live datasets, further demonstrating its practical applicability.

Submit your project link here:
```
https://github.com/Apostu0Alexandru/html-similarity
```

*Developed for Veridion internship application - March 2025*

---

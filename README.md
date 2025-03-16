# HTML Document Similarity Analyzer

A sophisticated algorithm that groups HTML documents based on how users perceive them in a web browser. This multi-dimensional analyzer evaluates content, visual appearance, DOM structure, and semantic meaning to identify similar webpages with high accuracy.

---

## 🌟 Project Overview

This project tackles the challenge of grouping HTML documents based on their similarity from a user's perspective. The key insight driving this solution is that users primarily perceive web pages through their visual appearance, content, and structure—not through raw HTML code.

By analyzing multiple dimensions of similarity, this algorithm effectively identifies pages that would appear similar to users browsing the web, even when their underlying HTML implementations differ.

---

## 🎯 Methodology: A Multi-Dimensional Approach

### Multiple Analysis Methods

The analyzer offers three distinct approaches:

1. **Weighted Analysis** (Default): Combines multiple similarity metrics with configurable weights
2. **Visual Analysis**: Uses screenshot capturing and OpenCV for image-based comparison
3. **Hybrid Analysis**: Integrates both approaches for maximum accuracy

### Similarity Dimensions

The weighted analysis method evaluates HTML documents across four key dimensions:

#### 1. Content Similarity (30%)
Users immediately notice what they can read on the page. The algorithm extracts visible text and applies TF-IDF vectorization with cosine similarity to identify documents with similar textual content.

#### 2. Visual Appearance (40%)
The most immediate aspect users perceive is how a page looks. The algorithm captures this through:
- **Color scheme analysis**: Extracting and comparing dominant colors
- **CSS property comparison**: Analyzing layout, fonts, and styling
- **Visual fingerprinting**: Creating simplified visual representations of page layouts

#### 3. DOM Structure (20%)
The structural organization of a page affects how users interact with it. The algorithm analyzes:
- **Tag frequencies**: Counting HTML elements to identify similar page structures
- **Hierarchical DOM analysis**: Comparing the tree-like organization of elements
- **Tag importance weighting**: Giving higher importance to structural elements like headers

#### 4. Semantic Meaning (10%)
Understanding the purpose of page sections enhances similarity detection. The algorithm identifies:
- **Semantic HTML5 elements**: Detecting `header`, `nav`, `main`, etc.
- **Above-the-fold content prioritization**: Focusing on what users see first
- **Navigation patterns**: Comparing menu structures and interaction points

### OpenCV-Based Visual Analysis

For the visual analysis method, the algorithm performs:

- **Screenshot capture** using Playwright's headless browser
- **Perceptual hashing** to generate compact visual fingerprints (pHash, dHash, wHash)
- **Histogram analysis** to compare brightness and color distributions
- **Edge detection** to identify structural elements using Canny algorithm
- **Color palette extraction** to compare dominant colors between pages

This approach works completely offline without requiring external API services.

---

## 🔍 The Algorithm in Action

The algorithm processes HTML documents in several steps:

1. **Feature Extraction**: Analyzes each HTML file to extract multi-dimensional features
2. **Similarity Computation**: Calculates similarity scores between all document pairs
3. **Threshold-Based Grouping**: Groups documents with similarity scores above the configured threshold
4. **Visualization**: Generates similarity matrices as heatmaps (optional)

---

## 💻 Usage Instructions

### Installation:

```bash
# Clone the repository
git clone https://github.com/Apostu0Alexandru/html-similarity.git
cd html-similarity

# Install required dependencies
pip install beautifulsoup4 sklearn numpy matplotlib pillow cssutils
pip install playwright opencv-python imagehash  # For visual analysis
python -m playwright install  # Install browser binaries
```

### Basic Usage:

```bash
# Weighted analysis (default)
python html_similarity.py /path/to/html/files

# Visual analysis with screenshots
python html_similarity.py /path/to/html/files --method visual --capture-screenshots

# Hybrid approach
python html_similarity.py /path/to/html/files --method hybrid --capture-screenshots

# Generate visualizations
python html_similarity.py /path/to/html/files --visualize

# Save results to JSON
python html_similarity.py /path/to/html/files --output results.json
```

### Advanced Options:

```bash
python html_similarity.py /path/to/html/files \
  --method weighted \
  --content-weight 0.3 \
  --structure-weight 0.2 \
  --visual-weight 0.4 \
  --semantic-weight 0.1 \
  --threshold 0.75 \
  --visualize \
  --output results.json
```

---

## 📊 Example Results and Visualization

For a directory containing HTML files, the output might look like:

```
Group 1: ['aemails.org.html', 'afro-pari.com.html', 'aevesdk3.com.html', 'aigner-haag.at.html']
Group 2: ['aitoka.shop.html']
```

With the `--visualize` option, the analyzer generates similarity matrix heatmaps:

- Brighter colors indicate higher similarity between documents
- Documents are ordered to highlight clusters of similar pages
- Separate heatmaps can be generated for each subdirectory

---

## 🛠️ Design Decisions and Technical Considerations

### Why OpenCV Instead of External APIs

After exploring external visual similarity APIs, I implemented a local OpenCV-based solution that:
- Works completely offline without authentication issues
- Uses perceptual hashing for efficient image comparison
- Provides more control over the analysis process
- Eliminates external dependencies and potential rate limits

### Reliability Through Multiple Methods

The analyzer's hybrid approach ensures reliable results by:
- Falling back to the weighted method if visual analysis fails
- Combining the strengths of text-based and visual-based analysis
- Allowing users to choose the method that best fits their needs

### Efficient Processing for Large Datasets

The solution is optimized for performance with:
- Selective feature extraction that focuses on relevant HTML elements
- Vectorized operations using NumPy for efficient calculations
- Progressive directory processing to manage memory usage
- Adjustable timeout settings for screenshot capture

---

## 🔮 Future Enhancements

1. **Interactive visualization dashboard** for exploring similarity relationships
2. **Adaptive weighting system** that automatically adjusts weights based on dataset characteristics
3. **Distributed processing** for extremely large datasets
4. **Deep learning-based feature extraction** for more nuanced visual comparison

---

## 🏁 Conclusion

This HTML similarity analyzer effectively groups documents based on how users perceive them in a web browser by combining content, visual appearance, structure, and semantics into a unified model. The implementation offers multiple analysis methods, with the OpenCV-based visual analysis providing a robust alternative to external API services.

```
https://github.com/Apostu0Alexandru/html-similarity
```

*Developed for Veridion internship application - March 2025*

---

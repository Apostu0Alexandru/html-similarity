import os
import re
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import argparse
import time
import json
import logging
import hashlib
from PIL import Image, ImageDraw, ImageStat
from io import BytesIO
from collections import Counter
import cssutils
import colorsys
from playwright.sync_api import sync_playwright
import cv2
import imagehash

# Suppress cssutils logging
cssutils.log.setLevel(logging.FATAL)

class HTMLSimilarityAnalyzer:
    """
    A comprehensive analyzer that groups similar HTML documents based on multiple similarity metrics.
    Uses local screenshot analysis rather than external API services.
    """
    
    def __init__(self, content_weight=0.3, structure_weight=0.2, visual_weight=0.4, 
                 semantic_weight=0.1, threshold=0.75, method="weighted", capture_screenshots=False):
        """
        Initialize the analyzer with weights for different similarity components.
        
        Args:
            content_weight: Weight for text content similarity (default: 0.3)
            structure_weight: Weight for DOM structure similarity (default: 0.2)
            visual_weight: Weight for visual appearance similarity (default: 0.4)
            semantic_weight: Weight for semantic element similarity (default: 0.1)
            threshold: Similarity threshold for grouping documents (default: 0.75)
            method: Analysis method to use - "weighted", "visual", or "hybrid" (default: "weighted")
            capture_screenshots: Whether to capture and analyze actual screenshots (default: False)
        """
        self.content_weight = content_weight
        self.structure_weight = structure_weight
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
        self.threshold = threshold
        self.method = method
        self.capture_screenshots = capture_screenshots
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Initialize playwright if capturing screenshots
        if self.capture_screenshots:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
    
    def extract_features(self, html_files, directory):
        """
        Extract multiple features from HTML files to capture various aspects of similarity.
        
        Args:
            html_files: List of HTML file names
            directory: Directory containing the HTML files
            
        Returns:
            Dictionary containing extracted features
        """
        print(f"Extracting features from {len(html_files)} HTML files...")
        
        features = {
            'file_names': html_files,
            'text_contents': [],
            'dom_structures': [],
            'tag_frequencies': [],
            'css_properties': [],
            'color_schemes': [],
            'semantic_structures': [],
            'visual_fingerprints': [],
            'above_fold_content': [],
            'screenshot_features': []  # New feature for local screenshot analysis
        }
        
        for html_file in html_files:
            file_path = os.path.join(directory, html_file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    html_content = file.read()
                    
                    # Parse HTML
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract text content
                    text_content = self._extract_visible_text(soup)
                    features['text_contents'].append(text_content)
                    
                    # Extract DOM structure
                    dom_structure = self._get_dom_structure(soup)
                    features['dom_structures'].append(dom_structure)
                    
                    # Extract tag frequencies
                    tag_freq = self._get_tag_frequency(soup)
                    features['tag_frequencies'].append(tag_freq)
                    
                    # Extract CSS properties
                    css_props = self._extract_css_properties(soup, html_content)
                    features['css_properties'].append(css_props)
                    
                    # Extract color scheme
                    colors = self._extract_color_scheme(soup, html_content)
                    features['color_schemes'].append(colors)
                    
                    # Extract semantic structure
                    semantic = self._extract_semantic_structure(soup)
                    features['semantic_structures'].append(semantic)
                    
                    # Generate visual fingerprint
                    visual = self._generate_visual_fingerprint(soup, html_content)
                    features['visual_fingerprints'].append(visual)
                    
                    # Extract above-fold content (what user sees first)
                    above_fold = self._extract_above_fold_content(soup)
                    features['above_fold_content'].append(above_fold)
                    
                    # Capture and analyze screenshot for visual features
                    if self.capture_screenshots:
                        screenshot_feature = self._extract_screenshot_features(file_path)
                        features['screenshot_features'].append(screenshot_feature)
                    else:
                        features['screenshot_features'].append(None)
                    
            except Exception as e:
                print(f"Error processing {html_file}: {e}")
                # Add empty features for failed files to maintain index alignment
                features['text_contents'].append("")
                features['dom_structures'].append({})
                features['tag_frequencies'].append({})
                features['css_properties'].append({})
                features['color_schemes'].append([])
                features['semantic_structures'].append({})
                features['visual_fingerprints'].append([])
                features['above_fold_content'].append("")
                features['screenshot_features'].append(None)
        
        return features
    
    def _extract_visible_text(self, soup):
        """Extract visible text content from HTML"""
        # Remove script, style, and other non-visible elements
        for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]']):
            element.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean the text (remove extra whitespace, etc.)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _get_dom_structure(self, soup):
        """Get DOM structure as a hierarchical dictionary"""
        def extract_structure(element):
            result = {'tag': element.name}
            
            # Get attributes
            if element.attrs:
                result['attrs'] = {}
                for key, value in element.attrs.items():
                    # Convert to string if it's a list or other non-string
                    if isinstance(value, list):
                        value = ' '.join(value)
                    result['attrs'][key] = str(value)
            
            # Get children
            children = [child for child in element.children if child.name is not None]
            if children:
                result['children'] = [extract_structure(child) for child in children]
            
            return result
        
        # Start from html tag
        html_tag = soup.find('html')
        if html_tag:
            return extract_structure(html_tag)
        return {}
    
    def _get_tag_frequency(self, soup):
        """Get frequency of HTML tags in the document"""
        all_tags = soup.find_all(True)
        tag_counts = {}
        
        for tag in all_tags:
            tag_name = tag.name.lower()
            if tag_name in tag_counts:
                tag_counts[tag_name] += 1
            else:
                tag_counts[tag_name] = 1
        
        return tag_counts
    
    def _extract_css_properties(self, soup, html_content):
        """Extract CSS properties that affect the visual appearance"""
        css_props = {
            'fonts': [],
            'background_colors': [],
            'text_colors': [],
            'layout_props': {}
        }
        
        # Extract inline styles
        elements_with_style = soup.select('[style]')
        for element in elements_with_style:
            style_text = element.get('style', '')
            try:
                style = cssutils.parseStyle(style_text)
                
                # Extract font information
                if style.getPropertyValue('font-family'):
                    css_props['fonts'].append(style.getPropertyValue('font-family'))
                
                # Extract background colors
                if style.getPropertyValue('background-color'):
                    css_props['background_colors'].append(style.getPropertyValue('background-color'))
                
                # Extract text colors
                if style.getPropertyValue('color'):
                    css_props['text_colors'].append(style.getPropertyValue('color'))
                
                # Extract layout properties
                for prop in ['display', 'position', 'float', 'flex', 'grid']:
                    if style.getPropertyValue(prop):
                        if prop not in css_props['layout_props']:
                            css_props['layout_props'][prop] = []
                        css_props['layout_props'][prop].append(style.getPropertyValue(prop))
            except:
                pass
        
        # Extract style tags
        style_tags = soup.find_all('style')
        for style_tag in style_tags:
            try:
                if style_tag.string:
                    sheet = cssutils.parseString(style_tag.string)
                    for rule in sheet:
                        if rule.type == rule.STYLE_RULE:
                            # Font information
                            if rule.style.getPropertyValue('font-family'):
                                css_props['fonts'].append(rule.style.getPropertyValue('font-family'))
                            
                            # Background colors
                            if rule.style.getPropertyValue('background-color'):
                                css_props['background_colors'].append(rule.style.getPropertyValue('background-color'))
                            
                            # Text colors
                            if rule.style.getPropertyValue('color'):
                                css_props['text_colors'].append(rule.style.getPropertyValue('color'))
                            
                            # Layout properties
                            for prop in ['display', 'position', 'float', 'flex', 'grid']:
                                if rule.style.getPropertyValue(prop):
                                    if prop not in css_props['layout_props']:
                                        css_props['layout_props'][prop] = []
                                    css_props['layout_props'][prop].append(rule.style.getPropertyValue(prop))
            except:
                pass
        
        # Count frequencies
        css_props['fonts'] = dict(Counter(css_props['fonts']))
        css_props['background_colors'] = dict(Counter(css_props['background_colors']))
        css_props['text_colors'] = dict(Counter(css_props['text_colors']))
        for prop in css_props['layout_props']:
            css_props['layout_props'][prop] = dict(Counter(css_props['layout_props'][prop]))
        
        return css_props
    
    def _extract_color_scheme(self, soup, html_content):
        """Extract the color scheme of the document"""
        # Collect all colors
        colors = []
        
        # From inline styles
        elements_with_color = soup.select('[style*="color"]')
        for element in elements_with_color:
            style_text = element.get('style', '')
            try:
                style = cssutils.parseStyle(style_text)
                if style.getPropertyValue('color'):
                    colors.append(style.getPropertyValue('color'))
                if style.getPropertyValue('background-color'):
                    colors.append(style.getPropertyValue('background-color'))
            except:
                pass
        
        # From style tags
        style_tags = soup.find_all('style')
        for style_tag in style_tags:
            try:
                if style_tag.string:
                    sheet = cssutils.parseString(style_tag.string)
                    for rule in sheet:
                        if rule.type == rule.STYLE_RULE:
                            if rule.style.getPropertyValue('color'):
                                colors.append(rule.style.getPropertyValue('color'))
                            if rule.style.getPropertyValue('background-color'):
                                colors.append(rule.style.getPropertyValue('background-color'))
            except:
                pass
        
        # Process collected colors
        processed_colors = []
        for color in colors:
            try:
                # Parse CSS color to RGB
                if color.startswith('#'):
                    # Hex color
                    if len(color) == 4:  # Short hex (#RGB)
                        r = int(color[1] + color[1], 16)
                        g = int(color[2] + color[2], 16)
                        b = int(color[3] + color[3], 16)
                    else:  # Full hex (#RRGGBB)
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                    processed_colors.append((r, g, b))
                elif color.startswith('rgb'):
                    # RGB or RGBA color
                    values = re.findall(r'\d+', color)
                    if len(values) >= 3:
                        r = int(values[0])
                        g = int(values[1])
                        b = int(values[2])
                        processed_colors.append((r, g, b))
            except:
                pass
        
        return processed_colors
    
    def _extract_semantic_structure(self, soup):
        """Extract semantic structure using HTML5 semantic elements"""
        semantic_tags = ['header', 'nav', 'main', 'article', 'section', 'aside', 'footer']
        semantic_structure = {}
        
        for tag_name in semantic_tags:
            elements = soup.find_all(tag_name)
            if elements:
                semantic_structure[tag_name] = len(elements)
                # Get hierarchical relationship
                for i, element in enumerate(elements):
                    parent_tags = [parent.name for parent in element.parents if parent.name in semantic_tags]
                    if parent_tags:
                        key = f"{tag_name}_{i}_parents"
                        semantic_structure[key] = parent_tags
        
        # If no semantic tags found, fall back to div structure
        if not semantic_structure:
            divs = soup.find_all('div', class_=True)
            for div in divs:
                class_name = " ".join(div.get('class', []))
                semantic_hints = ['header', 'nav', 'main', 'content', 'article', 'section', 'aside', 'footer', 'menu']
                for hint in semantic_hints:
                    if hint in class_name.lower():
                        if hint not in semantic_structure:
                            semantic_structure[hint] = 0
                        semantic_structure[hint] += 1
        
        return semantic_structure
    
    def _generate_visual_fingerprint(self, soup, html_content):
        """Generate a simplified visual fingerprint based on DOM structure"""
        # Create a simplified 10x10 grid representation of the page
        grid_size = 10
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Find all visible elements with size/position info
        visible_elements = []
        for element in soup.find_all(True):
            # Skip invisible elements
            if element.name in ['script', 'style', 'meta', 'link', 'noscript']:
                continue
                
            # Get element attributes
            style = element.get('style', '')
            classes = element.get('class', [])
            element_id = element.get('id', '')
            
            # Rough visibility score based on element attributes
            visibility_score = 0
            
            # Important elements are more visible
            if element.name in ['h1', 'h2', 'h3', 'img', 'header', 'footer']:
                visibility_score += 3
            elif element.name in ['p', 'a', 'div', 'span', 'li']:
                visibility_score += 1
                
            # Check if element has content
            if element.text.strip() or element.name == 'img':
                visibility_score += 1
            
            # Check for visibility hints in style
            if 'display:none' in style or 'visibility:hidden' in style:
                visibility_score = 0
            
            if visibility_score > 0:
                # Approximate position in the document
                position = len(visible_elements) / 100  # Normalize to 0-1 range
                size = len(element.text) if element.text else 1
                size = min(1.0, size / 500)  # Normalize to 0-1 range
                
                visible_elements.append((element.name, position, size, visibility_score))
        
        # Map elements to the grid
        for element in visible_elements:
            name, position, size, visibility = element
            
            # Map position to grid coordinates
            row = int(position * grid_size)
            row = min(row, grid_size - 1)  # Ensure within bounds
            
            # Determine column based on element type
            if name in ['h1', 'h2', 'h3', 'header', 'footer']:
                col_start = 0
                col_end = grid_size - 1
            elif name == 'img':
                col_start = int(grid_size / 4)
                col_end = int(3 * grid_size / 4)
            else:
                col_start = int(grid_size / 6)
                col_end = int(5 * grid_size / 6)
            
            # Set visibility in the grid
            for col in range(col_start, col_end + 1):
                grid[row][col] = max(grid[row][col], visibility)
        
        # Flatten the grid to a 1D fingerprint
        fingerprint = [cell for row in grid for cell in row]
        
        return fingerprint
    
    def _extract_above_fold_content(self, soup):
        """Extract content likely to be visible above the fold (first screen)"""
        above_fold = []
        
        # Headers are usually above the fold
        for tag in ['h1', 'h2', 'h3']:
            elements = soup.find_all(tag)
            for element in elements[:2]:  # Take first two headers of each type
                above_fold.append(element.get_text().strip())
        
        # First few paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs[:3]:  # Take first three paragraphs
            above_fold.append(p.get_text().strip())
        
        # First image alt text
        images = soup.find_all('img')
        for img in images[:2]:  # Take first two images
            alt_text = img.get('alt', '')
            if alt_text:
                above_fold.append(alt_text)
        
        # Main navigation
        nav = soup.find('nav')
        if nav:
            nav_text = nav.get_text().strip()
            above_fold.append(nav_text)
        
        # Join all content
        return ' '.join(above_fold)
    
    def _extract_screenshot_features(self, html_path):
        """
        Extract visual features from a screenshot of the HTML file.
        Uses perceptual hashing and other local image features instead of external API.
        
        Args:
            html_path: Path to the HTML file
            
        Returns:
            Dictionary of screenshot features or None if screenshot failed
        """
        try:
            # Capture screenshot
            screenshot = self._capture_screenshot(html_path)
            if screenshot is None:
                return None
            
            # Generate features from screenshot
            features = {}
            
            # 1. Compute perceptual hashes
            features['phash'] = str(imagehash.phash(screenshot))
            features['dhash'] = str(imagehash.dhash(screenshot))
            features['whash'] = str(imagehash.whash(screenshot))
            
            # 2. Convert to numpy array for OpenCV processing
            img_array = np.array(screenshot)
            if len(img_array.shape) == 3:  # Color image
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:  # Already grayscale
                img_gray = img_array
            
            # 3. Calculate histograms
            hist = cv2.calcHist([img_gray], [0], None, [32], [0, 256])
            features['histogram'] = hist.flatten().tolist()
            
            # 4. Calculate image statistics
            stat = ImageStat.Stat(screenshot)
            features['mean'] = stat.mean
            features['median'] = stat.median
            features['stddev'] = stat.stddev
            
            # 5. Extract dominant colors (simplified)
            if screenshot.mode == 'RGB':
                # Resize to smaller image for faster processing
                small_img = screenshot.resize((50, 50))
                pixels = small_img.getdata()
                colors = Counter(pixels)
                top_colors = colors.most_common(5)
                features['dominant_colors'] = [list(color[0]) for color in top_colors]
            else:
                features['dominant_colors'] = []
            
            # 6. Edge detection
            edges = cv2.Canny(img_gray, 100, 200)
            edge_points = np.sum(edges > 0)
            features['edge_density'] = float(edge_points) / (edges.shape[0] * edges.shape[1])
            
            return features
        
        except Exception as e:
            print(f"Error extracting screenshot features: {e}")
            return None
    
    def _capture_screenshot(self, html_path):
        """
        Capture screenshot of an HTML file using Playwright.
        
        Args:
            html_path: Path to the HTML file
            
        Returns:
            PIL Image object or None if capture failed
        """
        if not self.capture_screenshots:
            return None
            
        try:
            # Get absolute path for file URL
            file_path = os.path.abspath(html_path)
            file_url = f"file://{file_path}"
            
            # Create a new page
            page = self.browser.new_page(viewport={"width": 1280, "height": 1024})
            
            # Set a reasonable timeout
            page.set_default_timeout(10000)  # 10 seconds
            
            # Load the page
            page.goto(file_url)
            page.wait_for_load_state("domcontentloaded")
            
            # Take screenshot
            screenshot_bytes = page.screenshot()
            
            # Close the page
            page.close()
            
            # Convert to PIL Image
            image = Image.open(BytesIO(screenshot_bytes))
            
            return image
        except Exception as e:
            print(f"Error capturing screenshot for {html_path}: {e}")
            return None
    
    def compute_similarity_matrix(self, features):
        """
        Compute combined similarity matrix based on multiple feature types.
        The method used depends on the analyzer's configuration.
        
        Args:
            features: Dictionary containing extracted features
            
        Returns:
            Combined similarity matrix
        """
        if self.method == "visual" and self.capture_screenshots:
            # Use visual-only similarity based on screenshots
            return self._compute_screenshot_similarity(features['screenshot_features'])
        else:
            # Use weighted combination of multiple similarity metrics
            print("Computing similarity matrix using multiple metrics...")
            
            # Content similarity
            content_similarity = self._compute_text_similarity(features['text_contents'])
            
            # Structure similarity
            structure_similarity = self._compute_structure_similarity(features['dom_structures'], features['tag_frequencies'])
            
            # Visual similarity
            if self.capture_screenshots:
                # Use screenshot-based visual similarity if available
                screenshot_similarity = self._compute_screenshot_similarity(features['screenshot_features'])
                visual_similarity = 0.6 * screenshot_similarity + 0.4 * self._compute_visual_similarity(
                    features['css_properties'], 
                    features['color_schemes']
                )
            else:
                # Fall back to CSS and color-based visual similarity
                visual_similarity = self._compute_visual_similarity(
                    features['css_properties'], 
                    features['color_schemes']
                )
            
            # Semantic similarity
            semantic_similarity = self._compute_semantic_similarity(
                features['semantic_structures'], 
                features['above_fold_content']
            )
            
            # Combine similarities with weights
            combined_similarity = (
                self.content_weight * content_similarity +
                self.structure_weight * structure_similarity +
                self.visual_weight * visual_similarity +
                self.semantic_weight * semantic_similarity
            )
            
            return combined_similarity
    
    def _compute_text_similarity(self, text_contents):
        """Compute similarity between text contents using TF-IDF"""
        # Handle empty text contents
        if not text_contents or all(not text for text in text_contents):
            return np.zeros((len(text_contents), len(text_contents)))
        
        # Replace empty strings with a placeholder
        processed_contents = [text if text else "empty_document" for text in text_contents]
        
        # Compute TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(processed_contents)
        
        # Compute cosine similarity
        text_similarity = cosine_similarity(tfidf_matrix)
        
        return text_similarity
    
    def _compute_structure_similarity(self, dom_structures, tag_frequencies):
        """Compute similarity between documents based on DOM structure and tag frequencies"""
        n = len(dom_structures)
        structure_similarity = np.zeros((n, n))
        
        # Compute similarity based on tag frequencies
        for i in range(n):
            for j in range(n):
                # Skip if either document has no structure
                if not dom_structures[i] or not dom_structures[j]:
                    structure_similarity[i, j] = 0
                    continue
                
                # Tag frequency similarity
                tag_sim = self._compute_tag_frequency_similarity(tag_frequencies[i], tag_frequencies[j])
                
                # DOM structure similarity
                dom_sim = self._compute_dom_similarity(dom_structures[i], dom_structures[j])
                
                # Weighted combination
                structure_similarity[i, j] = 0.4 * tag_sim + 0.6 * dom_sim
        
        return structure_similarity
    
    def _compute_tag_frequency_similarity(self, tags1, tags2):
        """Compute similarity between tag frequency dictionaries"""
        # If both dictionaries are empty, consider them similar
        if not tags1 and not tags2:
            return 1.0
        
        # If one dictionary is empty, consider them dissimilar
        if not tags1 or not tags2:
            return 0.0
        
        # Get all unique tags
        all_tags = set(tags1.keys()) | set(tags2.keys())
        
        # Convert to vectors
        vector1 = [tags1.get(tag, 0) for tag in all_tags]
        vector2 = [tags2.get(tag, 0) for tag in all_tags]
        
        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(a * a for a in vector1) ** 0.5
        norm2 = sum(b * b for b in vector2) ** 0.5
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0
    
    def _compute_dom_similarity(self, dom1, dom2):
        """Compute similarity between DOM structures"""
        def compare_trees(tree1, tree2, depth=0):
            # Base similarity for matching tag
            if tree1['tag'] == tree2['tag']:
                base_similarity = 1.0
            else:
                # Some tags are more similar than others
                similar_tags = {
                    'div': ['section', 'article', 'main'],
                    'h1': ['h2', 'h3'],
                    'h2': ['h1', 'h3'],
                    'h3': ['h1', 'h2', 'h4'],
                    'ul': ['ol', 'menu'],
                    'ol': ['ul', 'menu'],
                    'span': ['p', 'div'],
                    'p': ['span', 'div']
                }
                if tree1['tag'] in similar_tags and tree2['tag'] in similar_tags.get(tree1['tag'], []):
                    base_similarity = 0.7
                else:
                    base_similarity = 0.0
            
            # Weight attributes similarity
            attr_similarity = 0.0
            if 'attrs' in tree1 and 'attrs' in tree2:
                # Count matching attributes
                shared_attrs = set(tree1['attrs'].keys()) & set(tree2['attrs'].keys())
                total_attrs = set(tree1['attrs'].keys()) | set(tree2['attrs'].keys())
                
                if total_attrs:
                    attr_similarity = len(shared_attrs) / len(total_attrs)
            
            # Compute children similarity
            children_similarity = 0.0
            if 'children' in tree1 and 'children' in tree2:
                # Get number of children to compare
                min_children = min(len(tree1['children']), len(tree2['children']))
                max_children = max(len(tree1['children']), len(tree2['children']))
                
                # Compare each child
                children_scores = []
                for i in range(min_children):
                    children_scores.append(compare_trees(
                        tree1['children'][i], 
                        tree2['children'][i],
                        depth + 1
                    ))
                
                # Average children similarity
                if children_scores:
                    avg_children_sim = sum(children_scores) / len(children_scores)
                    # Penalty for different number of children
                    children_count_ratio = min_children / max_children if max_children > 0 else 1.0
                    children_similarity = avg_children_sim * children_count_ratio
            
            # Weighted sum of similarities
            # Structure becomes more important deeper in the tree
            if depth == 0:
                return 0.5 * base_similarity + 0.2 * attr_similarity + 0.3 * children_similarity
            else:
                return 0.3 * base_similarity + 0.1 * attr_similarity + 0.6 * children_similarity
        
        # Compare the DOM trees starting from the root
        return compare_trees(dom1, dom2)
    
    def _compute_visual_similarity(self, css_properties, color_schemes):
        """Compute visual similarity between documents based on CSS and colors"""
        n = len(css_properties)
        visual_similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # CSS properties similarity
                css_sim = self._compute_css_similarity(css_properties[i], css_properties[j])
                
                # Color scheme similarity
                color_sim = self._compute_color_similarity(color_schemes[i], color_schemes[j])
                
                # Weighted combination
                visual_similarity[i, j] = 0.5 * css_sim + 0.5 * color_sim
        
        return visual_similarity
    
    def _compute_screenshot_similarity(self, screenshot_features):
        """
        Compute similarity between screenshots using local image features.
        This is the alternative to using external APIs like Twelve Labs.
        
        Args:
            screenshot_features: List of screenshot feature dictionaries
            
        Returns:
            Screenshot similarity matrix
        """
        n = len(screenshot_features)
        screenshot_similarity = np.ones((n, n))  # Default to identity matrix
        
        # Skip if no screenshot features available
        valid_features = [f for f in screenshot_features if f is not None]
        if not valid_features:
            return screenshot_similarity
        
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle (symmetric matrix)
                # Skip if either screenshot has no features
                if screenshot_features[i] is None or screenshot_features[j] is None:
                    screenshot_similarity[i, j] = 0.0
                    screenshot_similarity[j, i] = 0.0
                    continue
                
                # Compute individual feature similarities
                similarities = []
                
                # 1. Perceptual hash similarity
                for hash_type in ['phash', 'dhash', 'whash']:
                    hash1 = screenshot_features[i][hash_type]
                    hash2 = screenshot_features[j][hash_type]
                    # Hamming distance between hashes (normalized)
                    hash_sim = 1.0 - sum(c1 != c2 for c1, c2 in zip(hash1, hash2)) / len(hash1)
                    similarities.append((hash_sim, 0.4))  # Higher weight for perceptual hashes
                
                # 2. Histogram similarity
                hist1 = np.array(screenshot_features[i]['histogram'])
                hist2 = np.array(screenshot_features[j]['histogram'])
                # Normalize histograms
                hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1
                hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2
                # Histogram intersection
                hist_sim = np.sum(np.minimum(hist1, hist2))
                similarities.append((hist_sim, 0.2))
                
                # 3. Color similarity
                if screenshot_features[i]['dominant_colors'] and screenshot_features[j]['dominant_colors']:
                    # Compare top dominant colors
                    color1 = np.array(screenshot_features[i]['dominant_colors'][0])
                    color2 = np.array(screenshot_features[j]['dominant_colors'][0])
                    # Color distance (normalized)
                    color_dist = np.linalg.norm(color1 - color2) / (255 * np.sqrt(3))
                    color_sim = 1.0 - color_dist
                    similarities.append((color_sim, 0.2))
                
                # 4. Structure similarity (edge density)
                edge_sim = 1.0 - abs(screenshot_features[i]['edge_density'] - screenshot_features[j]['edge_density'])
                similarities.append((edge_sim, 0.2))
                
                # Compute weighted average similarity
                total_weight = sum(weight for _, weight in similarities)
                if total_weight > 0:
                    avg_sim = sum(sim * weight for sim, weight in similarities) / total_weight
                    screenshot_similarity[i, j] = avg_sim
                    screenshot_similarity[j, i] = avg_sim  # Symmetric matrix
                
        return screenshot_similarity
    
    def _compute_css_similarity(self, css1, css2):
        """Compute similarity between CSS properties"""
        if not css1 or not css2:
            return 0.0
        
        similarities = []
        
        # Compare fonts
        if css1['fonts'] and css2['fonts']:
            fonts1 = set(css1['fonts'].keys())
            fonts2 = set(css2['fonts'].keys())
            if fonts1 and fonts2:
                font_similarity = len(fonts1 & fonts2) / len(fonts1 | fonts2)
                similarities.append(font_similarity)
        
        # Compare background colors
        if css1['background_colors'] and css2['background_colors']:
            bg_colors1 = set(css1['background_colors'].keys())
            bg_colors2 = set(css2['background_colors'].keys())
            if bg_colors1 and bg_colors2:
                bg_color_similarity = len(bg_colors1 & bg_colors2) / len(bg_colors1 | bg_colors2)
                similarities.append(bg_color_similarity)
        
        # Compare text colors
        if css1['text_colors'] and css2['text_colors']:
            text_colors1 = set(css1['text_colors'].keys())
            text_colors2 = set(css2['text_colors'].keys())
            if text_colors1 and text_colors2:
                text_color_similarity = len(text_colors1 & text_colors2) / len(text_colors1 | text_colors2)
                similarities.append(text_color_similarity)
        
        # Compare layout properties
        layout_similarities = []
        for prop in set(css1['layout_props'].keys()) & set(css2['layout_props'].keys()):
            prop1 = set(css1['layout_props'][prop].keys())
            prop2 = set(css2['layout_props'][prop].keys())
            if prop1 and prop2:
                prop_similarity = len(prop1 & prop2) / len(prop1 | prop2)
                layout_similarities.append(prop_similarity)
        
        if layout_similarities:
            similarities.append(sum(layout_similarities) / len(layout_similarities))
        
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    
    def _compute_color_similarity(self, colors1, colors2):
        """Compute similarity between color schemes"""
        if not colors1 or not colors2:
            return 0.0
        
        # Convert RGB to HSV for better color comparison
        hsv1 = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in colors1]
        hsv2 = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in colors2]
        
        # Calculate average hue, saturation, value
        avg_hsv1 = [sum(x)/len(hsv1) for x in zip(*hsv1)] if hsv1 else [0, 0, 0]
        avg_hsv2 = [sum(x)/len(hsv2) for x in zip(*hsv2)] if hsv2 else [0, 0, 0]
        
        # Calculate difference in hue, saturation, value
        hue_diff = min(abs(avg_hsv1[0] - avg_hsv2[0]), 1 - abs(avg_hsv1[0] - avg_hsv2[0]))
        sat_diff = abs(avg_hsv1[1] - avg_hsv2[1])
        val_diff = abs(avg_hsv1[2] - avg_hsv2[2])
        
        # Calculate similarity (1 - normalized difference)
        hue_similarity = 1 - hue_diff
        sat_similarity = 1 - sat_diff
        val_similarity = 1 - val_diff
        
        # Weighted color similarity
        color_similarity = 0.5 * hue_similarity + 0.25 * sat_similarity + 0.25 * val_similarity
        
        return color_similarity
    
    def _compute_semantic_similarity(self, semantic_structures, above_fold_contents):
        """Compute semantic similarity between documents"""
        n = len(semantic_structures)
        semantic_similarity = np.zeros((n, n))
        
        # Compute similarity based on semantic structures
        for i in range(n):
            for j in range(n):
                # Semantic structure similarity
                struct_sim = self._compute_semantic_structure_similarity(
                    semantic_structures[i], 
                    semantic_structures[j]
                )
                
                # Above-fold content similarity
                fold_sim = self._compute_text_fold_similarity(
                    above_fold_contents[i], 
                    above_fold_contents[j]
                )
                
                # Weighted combination
                semantic_similarity[i, j] = 0.6 * struct_sim + 0.4 * fold_sim
        
        return semantic_similarity
    
    def _compute_semantic_structure_similarity(self, struct1, struct2):
        """Compute similarity between semantic structures"""
        if not struct1 and not struct2:
            return 1.0
        
        if not struct1 or not struct2:
            return 0.0
        
        # Get common semantic elements
        common_elements = set(struct1.keys()) & set(struct2.keys())
        all_elements = set(struct1.keys()) | set(struct2.keys())
        
        # If no common elements, return 0
        if not common_elements:
            return 0.0
        
        # Calculate similarity for common elements
        element_similarities = []
        for element in common_elements:
            # Skip parent relationship elements
            if '_parents' in element:
                continue
            
            # Compare count of elements
            count1 = struct1.get(element, 0)
            count2 = struct2.get(element, 0)
            
            if count1 > 0 and count2 > 0:
                count_ratio = min(count1, count2) / max(count1, count2)
                element_similarities.append(count_ratio)
        
        # Calculate parent relationship similarities
        parent_similarities = []
        for element in common_elements:
            if '_parents' in element:
                parents1 = struct1.get(element, [])
                parents2 = struct2.get(element, [])
                
                if parents1 and parents2:
                    common_parents = set(parents1) & set(parents2)
                    all_parents = set(parents1) | set(parents2)
                    parent_sim = len(common_parents) / len(all_parents)
                    parent_similarities.append(parent_sim)
        
        # Combine element and parent similarities
        all_similarities = element_similarities + parent_similarities
        if all_similarities:
            avg_similarity = sum(all_similarities) / len(all_similarities)
            # Adjust by coverage ratio
            coverage_ratio = len(common_elements) / len(all_elements)
            return avg_similarity * coverage_ratio
        else:
            return 0.0
    
    def _compute_text_fold_similarity(self, text1, text2):
        """Compute similarity between above-fold content texts"""
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # Use TF-IDF with cosine similarity
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix)[0, 1]
        
        return similarity
    
    def group_similar_documents(self, similarity_matrix, file_names):
        """Group similar documents based on similarity matrix"""
        print(f"Grouping documents with similarity threshold {self.threshold}...")
        
        n = len(file_names)
        grouped = []
        processed = set()
        
        for i in range(n):
            if i in processed:
                continue
            
            group = [file_names[i]]
            processed.add(i)
            
            for j in range(i+1, n):
                if j not in processed and similarity_matrix[i, j] >= self.threshold:
                    group.append(file_names[j])
                    processed.add(j)
            
            grouped.append(group)
        
        print(f"Found {len(grouped)} groups of similar documents.")
        return grouped
    
    def visualize_similarity_matrix(self, similarity_matrix, file_names, output_path=None):
        """Visualize the similarity matrix as a heatmap"""
        plt.figure(figsize=(12, 10))
        plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Similarity')
        
        # Set ticks and labels
        plt.xticks(range(len(file_names)), [os.path.basename(f) for f in file_names], rotation=90)
        plt.yticks(range(len(file_names)), [os.path.basename(f) for f in file_names])
        
        plt.title('Document Similarity Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Similarity matrix visualization saved to {output_path}")
        else:
            plt.show()
    
    def analyze_directory(self, directory):
        """Analyze all HTML files in a directory and group similar documents"""
        # Get all HTML files in the directory
        html_files = [f for f in os.listdir(directory) if f.endswith('.html')]
        
        if not html_files:
            print(f"No HTML files found in {directory}")
            return [], None, []
        
        print(f"Found {len(html_files)} HTML files in {directory}")
        
        # Extract features from HTML files
        features = self.extract_features(html_files, directory)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(features)
        
        # Group similar documents
        groups = self.group_similar_documents(similarity_matrix, html_files)
        
        return groups, similarity_matrix, html_files
    
    def cleanup(self):
        """Clean up resources."""
        if self.capture_screenshots:
            self.browser.close()
            self.playwright.stop()


def main():
    """Main function to run the HTML similarity analyzer"""
    parser = argparse.ArgumentParser(description='Group similar HTML documents using multiple approaches')
    parser.add_argument('directory', help='Directory containing HTML files')
    parser.add_argument('--method', choices=['weighted', 'visual', 'hybrid'], default='weighted',
                        help='Analysis method to use (default: weighted)')
    parser.add_argument('--content-weight', type=float, default=0.3,
                        help='Weight for content-based similarity (default: 0.3)')
    parser.add_argument('--structure-weight', type=float, default=0.2,
                        help='Weight for structure-based similarity (default: 0.2)')
    parser.add_argument('--visual-weight', type=float, default=0.4,
                        help='Weight for visual appearance similarity (default: 0.4)')
    parser.add_argument('--semantic-weight', type=float, default=0.1,
                        help='Weight for semantic element similarity (default: 0.1)')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Similarity threshold for grouping documents (default: 0.75)')
    parser.add_argument('--capture-screenshots', action='store_true',
                        help='Enable screenshot capturing for visual comparison')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize similarity matrix')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HTMLSimilarityAnalyzer(
        content_weight=args.content_weight,
        structure_weight=args.structure_weight,
        visual_weight=args.visual_weight,
        semantic_weight=args.semantic_weight,
        threshold=args.threshold,
        method=args.method,
        capture_screenshots=args.capture_screenshots
    )
    
    try:
        # Process all subdirectories if the directory contains subdirectories
        root_dir = Path(args.directory)
        subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            print(f"Found {len(subdirs)} subdirectories to process")
            all_results = {}
            
            for subdir in subdirs:
                print(f"\nProcessing subdirectory: {subdir}")
                groups, similarity_matrix, html_files = analyzer.analyze_directory(subdir)
                
                print(f"Results for {subdir}:")
                for i, group in enumerate(groups):
                    print(f"Group {i+1}: {group}")
                
                all_results[subdir.name] = [group for group in groups]
                
                if args.visualize and similarity_matrix is not None:
                    output_path = None
                    if args.output:
                        vis_dir = Path(args.output).parent / 'visualizations'
                        os.makedirs(vis_dir, exist_ok=True)
                        output_path = vis_dir / f"{subdir.name}_{args.method}_similarity.png"
                    
                    analyzer.visualize_similarity_matrix(similarity_matrix, html_files, output_path)
            
            # Save results to JSON if output is specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Results saved to {args.output}")
        
        else:
            # Process the directory directly
            groups, similarity_matrix, html_files = analyzer.analyze_directory(args.directory)
            
            print("\nFinal results:")
            for i, group in enumerate(groups):
                print(f"Group {i+1}: {group}")
            
            if args.visualize and similarity_matrix is not None:
                output_path = None
                if args.output:
                    vis_dir = Path(args.output).parent / 'visualizations'
                    os.makedirs(vis_dir, exist_ok=True)
                    output_path = vis_dir / f"{args.method}_similarity.png"
                
                analyzer.visualize_similarity_matrix(similarity_matrix, html_files, output_path)
            
            # Save results to JSON if output is specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([group for group in groups], f, indent=2)
                print(f"Results saved to {args.output}")
    
    finally:
        # Clean up resources
        analyzer.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import json
import os
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import random

app = Flask(__name__)

# Configuration
VLM_DATA_DIR = "/home/colligo/project/vlm/Bagel/datasets/bagel_example/vlm"
JSONL_FILE = os.path.join(VLM_DATA_DIR, "llava_ov_si.jsonl")
IMAGES_DIR = os.path.join(VLM_DATA_DIR, "images")

class VLMDataLoader:
    def __init__(self):
        self.data = []
        self.load_data()
    
    def load_data(self):
        """Load VLM data from JSONL file"""
        if os.path.exists(JSONL_FILE):
            with open(JSONL_FILE, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            print(f"Loaded {len(self.data)} samples")
        else:
            print(f"JSONL file not found: {JSONL_FILE}")
    
    def get_sample(self, index):
        """Get a specific sample by index"""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None
    
    def get_random_sample(self):
        """Get a random sample"""
        if self.data:
            return random.choice(self.data)
        return None
    
    def get_total_count(self):
        """Get total number of samples"""
        return len(self.data)

# Initialize data loader
data_loader = VLMDataLoader()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', total_samples=data_loader.get_total_count())

@app.route('/api/sample/<int:index>')
def get_sample(index):
    """Get sample by index"""
    sample = data_loader.get_sample(index)
    if sample:
        return jsonify(sample)
    return jsonify({"error": "Sample not found"}), 404

@app.route('/api/random')
def get_random():
    """Get random sample"""
    sample = data_loader.get_random_sample()
    if sample:
        return jsonify(sample)
    return jsonify({"error": "No data available"}), 404

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    if not data_loader.data:
        return jsonify({"error": "No data loaded"})
    
    # Calculate basic stats
    total = len(data_loader.data)
    has_images = sum(1 for item in data_loader.data if 'image' in item)
    conversation_lengths = [len(item.get('conversations', [])) for item in data_loader.data]
    avg_conv_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
    
    stats = {
        "total_samples": total,
        "samples_with_images": has_images,
        "average_conversation_length": round(avg_conv_length, 2),
        "min_conversation_length": min(conversation_lengths) if conversation_lengths else 0,
        "max_conversation_length": max(conversation_lengths) if conversation_lengths else 0
    }
    
    return jsonify(stats)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the VLM dataset"""
    return send_from_directory(IMAGES_DIR, filename)

# Create templates directory and HTML template
if __name__ == '__main__':
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Dataset Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .sample-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .image-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .conversation-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .sample-image {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            display: block;
            margin: 0 auto;
        }
        
        .conversation {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        
        .human {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .gpt {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2196f3;
        }
        
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        
        button {
            padding: 10px 20px;
            background-color: #2196f3;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        
        button:hover {
            background-color: #1976d2;
        }
        
        input[type="number"] {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 100px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background-color: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .sample-container {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 VLM Dataset Viewer</h1>
        <p>Browse Vision-Language Model training data</p>
        <p><strong>Total Samples:</strong> {{ total_samples }}</p>
    </div>
    
    <div class="stats" id="stats">
        <h3>Dataset Statistics</h3>
        <div class="loading">Loading statistics...</div>
    </div>
    
    <div class="controls">
        <label>Sample Index:</label>
        <input type="number" id="sampleIndex" min="0" max="{{ total_samples - 1 }}" value="0">
        <button onclick="loadSample()">Load Sample</button>
        <button onclick="loadRandom()">Random Sample</button>
        <button onclick="previousSample()">Previous</button>
        <button onclick="nextSample()">Next</button>
    </div>
    
    <div class="sample-container">
        <div class="image-panel">
            <h3>Image</h3>
            <div id="imageContainer">
                <div class="loading">Click "Load Sample" to view data</div>
            </div>
        </div>
        
        <div class="conversation-panel">
            <h3>Conversation</h3>
            <div id="conversationContainer">
                <div class="loading">Click "Load Sample" to view conversation</div>
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        const maxIndex = {{ total_samples - 1 }};
        
        // Load statistics on page load
        loadStats();
        
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('stats').innerHTML = '<div class="error">' + data.error + '</div>';
                        return;
                    }
                    
                    const statsHtml = `
                        <h3>Dataset Statistics</h3>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-value">${data.total_samples}</div>
                                <div class="stat-label">Total Samples</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${data.samples_with_images}</div>
                                <div class="stat-label">With Images</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${data.average_conversation_length}</div>
                                <div class="stat-label">Avg Conversation Length</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${data.min_conversation_length} - ${data.max_conversation_length}</div>
                                <div class="stat-label">Conv Length Range</div>
                            </div>
                        </div>
                    `;
                    document.getElementById('stats').innerHTML = statsHtml;
                })
                .catch(error => {
                    document.getElementById('stats').innerHTML = '<div class="error">Failed to load statistics</div>';
                });
        }
        
        function loadSample() {
            const index = parseInt(document.getElementById('sampleIndex').value);
            if (index < 0 || index > maxIndex) {
                alert('Invalid index. Please enter a number between 0 and ' + maxIndex);
                return;
            }
            
            currentIndex = index;
            
            document.getElementById('imageContainer').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('conversationContainer').innerHTML = '<div class="loading">Loading...</div>';
            
            fetch(`/api/sample/${index}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('imageContainer').innerHTML = '<div class="error">' + data.error + '</div>';
                        document.getElementById('conversationContainer').innerHTML = '<div class="error">' + data.error + '</div>';
                        return;
                    }
                    
                    displaySample(data);
                })
                .catch(error => {
                    document.getElementById('imageContainer').innerHTML = '<div class="error">Failed to load sample</div>';
                    document.getElementById('conversationContainer').innerHTML = '<div class="error">Failed to load sample</div>';
                });
        }
        
        function loadRandom() {
            document.getElementById('imageContainer').innerHTML = '<div class="loading">Loading...</div>';
            document.getElementById('conversationContainer').innerHTML = '<div class="loading">Loading...</div>';
            
            fetch('/api/random')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('imageContainer').innerHTML = '<div class="error">' + data.error + '</div>';
                        document.getElementById('conversationContainer').innerHTML = '<div class="error">' + data.error + '</div>';
                        return;
                    }
                    
                    // Update the index input to show current sample
                    currentIndex = data.id || 0;
                    document.getElementById('sampleIndex').value = currentIndex;
                    
                    displaySample(data);
                })
                .catch(error => {
                    document.getElementById('imageContainer').innerHTML = '<div class="error">Failed to load random sample</div>';
                    document.getElementById('conversationContainer').innerHTML = '<div class="error">Failed to load random sample</div>';
                });
        }
        
        function previousSample() {
            if (currentIndex > 0) {
                currentIndex--;
                document.getElementById('sampleIndex').value = currentIndex;
                loadSample();
            }
        }
        
        function nextSample() {
            if (currentIndex < maxIndex) {
                currentIndex++;
                document.getElementById('sampleIndex').value = currentIndex;
                loadSample();
            }
        }
        
        function displaySample(data) {
            // Display image
            const imageContainer = document.getElementById('imageContainer');
            if (data.image) {
                const imagePath = Array.isArray(data.image) ? data.image[0] : data.image;
                imageContainer.innerHTML = `
                    <p><strong>Image:</strong> ${imagePath}</p>
                    <img src="/images/${imagePath}" alt="Sample image" class="sample-image" 
                         onerror="this.onerror=null; this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4='; this.alt='Image not found';">
                `;
            } else {
                imageContainer.innerHTML = '<p>No image available</p>';
            }
            
            // Display conversation
            const conversationContainer = document.getElementById('conversationContainer');
            if (data.conversations && Array.isArray(data.conversations)) {
                let conversationHtml = '<div class="conversation">';
                
                data.conversations.forEach(conv => {
                    const messageClass = conv.from === 'human' ? 'human' : 'gpt';
                    const role = conv.from === 'human' ? '👤 Human' : '🤖 Assistant';
                    const value = conv.value || '';
                    
                    conversationHtml += `
                        <div class="message ${messageClass}">
                            <strong>${role}:</strong><br>
                            ${value.replace(/\\n/g, '<br>').replace(/</g, '&lt;').replace(/>/g, '&gt;')}
                        </div>
                    `;
                });
                
                conversationHtml += '</div>';
                
                // Add metadata
                conversationHtml += '<div style="margin-top: 15px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">';
                conversationHtml += `<strong>Sample ID:</strong> ${data.id || 'N/A'}<br>`;
                if (data.source) {
                    conversationHtml += `<strong>Source:</strong> ${data.source}<br>`;
                }
                conversationHtml += `<strong>Conversation Length:</strong> ${data.conversations.length} turns`;
                conversationHtml += '</div>';
                
                conversationContainer.innerHTML = conversationHtml;
            } else {
                conversationContainer.innerHTML = '<div class="error">No conversation data available</div>';
            }
        }
        
        // Allow Enter key to load sample
        document.getElementById('sampleIndex').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                loadSample();
            }
        });
    </script>
</body>
</html>"""
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(html_template)
    
    print("Starting VLM Dataset Viewer...")
    print("Access at: http://127.0.0.1:5000")
    print(f"Dataset: {VLM_DATA_DIR}")
    print(f"Total samples: {data_loader.get_total_count()}")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
from flask import Flask, request, jsonify
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)

# Configuration
INDEX_DIR = "rag_store"
DATA_PATH = "data/jenkins_data.jsonl"

# Global variables for models (load once)
index = None
metadata = None
embedder = None
tokenizer = None
llm = None


def load_system():
    """Load all ML models and indices"""
    global index, metadata, embedder, tokenizer, llm
    
    print("Loading FAISS index...")
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")

    print("Loading metadata...")
    with open(f"{INDEX_DIR}/metadata.json") as f:
        metadata = json.load(f)

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading LLM...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("✅ All models loaded successfully!")


def retrieve(query, k=5):
    """Retrieve top-k relevant documents"""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    dist, ids = index.search(q_emb, k)
    return [metadata[i] for i in ids[0]], dist[0]


def generate_answer(context, question):
    """Generate answer using LLM"""
    prompt = f"""You are an expert Jenkins CI/CD analyst.
Use ONLY the JSON data provided below.

JSON Data:
{context}

Question:
{question}

Your Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    output = llm.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the answer part (after "Your Answer:")
    if "Your Answer:" in answer:
        answer = answer.split("Your Answer:")[-1].strip()
    
    return answer


def get_build_stats():
    """Get statistics about builds"""
    stats = {
        'total_builds': len(metadata),
        'success': sum(1 for m in metadata if m['result'] == 'SUCCESS'),
        'failure': sum(1 for m in metadata if m['result'] == 'FAILURE'),
        'aborted': sum(1 for m in metadata if m['result'] == 'ABORTED'),
    }
    stats['success_rate'] = round((stats['success'] / stats['total_builds']) * 100, 2) if stats['total_builds'] > 0 else 0
    return stats


# ==================== HTML PAGE ====================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jenkins RAG Query System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; color: white; margin-bottom: 30px; }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            min-width: 120px;
            text-align: center;
        }
        .stat-card .number { font-size: 2em; font-weight: bold; display: block; }
        .stat-card .label { font-size: 0.9em; opacity: 0.9; }
        .main-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .input-group { display: flex; gap: 10px; margin-bottom: 20px; }
        #questionInput {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
        }
        #questionInput:focus { outline: none; border-color: #667eea; }
        #queryBtn {
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        #queryBtn:hover { transform: translateY(-2px); }
        #queryBtn:disabled { opacity: 0.6; cursor: not-allowed; }
        .loading { display: none; text-align: center; padding: 20px; color: #667eea; }
        .loading.active { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results { display: none; }
        .results.active { display: block; }
        .answer-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }
        .answer-section h3 { color: #667eea; margin-bottom: 15px; }
        .answer-text { line-height: 1.8; color: #333; white-space: pre-wrap; }
        .documents-section h3 { color: #333; margin-bottom: 20px; }
        .document-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .build-info { display: flex; gap: 15px; align-items: center; }
        .build-number { font-size: 1.2em; font-weight: bold; color: #333; }
        .badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-failure { background: #f8d7da; color: #721c24; }
        .badge-aborted { background: #fff3cd; color: #856404; }
        .doc-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #666;
        }
        .doc-text {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            color: #333;
        }
        .sample-queries {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .sample-queries h4 { color: #667eea; margin-bottom: 10px; }
        .sample-query {
            display: inline-block;
            background: white;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            border: 1px solid #e0e0e0;
        }
        .sample-query:hover { background: #667eea; color: white; }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Jenkins RAG Query System</h1>
            <p>Ask questions about your Jenkins CI/CD builds</p>
            <div class="stats-bar" id="stats"></div>
        </header>

        <div class="main-content">
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="Ask me anything about Jenkins builds..." onkeypress="handleKeyPress(event)">
                <button id="queryBtn" onclick="submitQuery()">Ask</button>
            </div>

            <div class="sample-queries">
                <h4>Sample Questions:</h4>
                <span class="sample-query" onclick="setSampleQuery('What was the result of the last build?')">What was the result of the last build?</span>
                <span class="sample-query" onclick="setSampleQuery('How many builds failed?')">How many builds failed?</span>
                <span class="sample-query" onclick="setSampleQuery('Which builds took the longest?')">Which builds took the longest?</span>
                <span class="sample-query" onclick="setSampleQuery('Show me successful builds')">Show me successful builds</span>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing your question...</p>
            </div>

            <div class="results" id="results">
                <div class="answer-section">
                    <h3>💡 Answer</h3>
                    <div class="answer-text" id="answer"></div>
                </div>
                <div class="documents-section">
                    <h3>📄 Related Build Information</h3>
                    <div id="documents"></div>
                </div>
            </div>

            <div class="error" id="error"></div>
        </div>
    </div>

    <script>
        // Load stats on page load
        fetch('/stats')
            .then(r => r.json())
            .then(stats => {
                document.getElementById('stats').innerHTML = `
                    <div class="stat-card"><span class="number">${stats.total_builds}</span><span class="label">Total Builds</span></div>
                    <div class="stat-card"><span class="number">${stats.success}</span><span class="label">Success</span></div>
                    <div class="stat-card"><span class="number">${stats.failure}</span><span class="label">Failures</span></div>
                    <div class="stat-card"><span class="number">${stats.aborted}</span><span class="label">Aborted</span></div>
                    <div class="stat-card"><span class="number">${stats.success_rate}%</span><span class="label">Success Rate</span></div>
                `;
            });

        function handleKeyPress(event) {
            if (event.key === 'Enter') submitQuery();
        }

        function setSampleQuery(query) {
            document.getElementById('questionInput').value = query;
            submitQuery();
        }

        async function submitQuery() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) { alert('Please enter a question'); return; }

            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');
            document.getElementById('error').style.display = 'none';
            document.getElementById('queryBtn').disabled = true;

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Something went wrong');

                document.getElementById('answer').textContent = data.answer;

                const docsHtml = data.documents.map(doc => `
                    <div class="document-card">
                        <div class="doc-header">
                            <div class="build-info">
                                <span class="build-number">Build #${doc.build_number}</span>
                                <span class="badge badge-${doc.result.toLowerCase()}">${doc.result}</span>
                            </div>
                        </div>
                        <div class="doc-details">
                            <div><strong>Commit:</strong> ${doc.commit}</div>
                            <div><strong>Duration:</strong> ${Math.round(doc.duration / 1000)}s</div>
                            <div><strong>Relevance:</strong> ${(1 / (1 + doc.distance)).toFixed(3)}</div>
                        </div>
                        <div class="doc-text">${doc.text}</div>
                        <a href="${doc.url}" target="_blank">View in Jenkins →</a>
                    </div>
                `).join('');

                document.getElementById('documents').innerHTML = docsHtml;
                document.getElementById('results').classList.add('active');

            } catch (error) {
                document.getElementById('error').textContent = `Error: ${error.message}`;
                document.getElementById('error').style.display = 'block';
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('queryBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Home page - returns HTML directly"""
    return HTML_PAGE


@app.route('/query', methods=['POST'])
def query():
    """Handle query requests"""
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        # Retrieve relevant documents
        retrieved, distances = retrieve(question, k=5)
        
        # Prepare context
        context = "\n\n".join([rec["text"] for rec in retrieved])
        
        # Generate answer
        answer = generate_answer(context, question)
        
        # Format retrieved documents
        docs = []
        for i, (rec, dist) in enumerate(zip(retrieved, distances)):
            docs.append({
                'rank': i + 1,
                'build_number': rec['build_number'],
                'result': rec['result'],
                'timestamp': rec['timestamp'],
                'duration': rec['duration'],
                'commit': rec['commit'][:8],
                'url': rec['url'],
                'distance': float(dist),
                'text': rec['text']
            })
        
        return jsonify({
            'answer': answer,
            'documents': docs,
            'context': context
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get build statistics"""
    return jsonify(get_build_stats())


@app.route('/builds')
def builds():
    """Get all builds"""
    builds_list = []
    for rec in metadata:
        builds_list.append({
            'build_number': rec['build_number'],
            'result': rec['result'],
            'timestamp': rec['timestamp'],
            'duration': rec['duration'],
            'commit': rec['commit'][:8],
            'url': rec['url']
        })
    builds_list.sort(key=lambda x: x['build_number'], reverse=True)
    return jsonify(builds_list)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([index is not None, metadata is not None, 
                             embedder is not None, tokenizer is not None, llm is not None])
    })


if __name__ == '__main__':
    # Load models before starting the server
    load_system()
    
    # Run Flask app
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("📊 Open your browser and visit: http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
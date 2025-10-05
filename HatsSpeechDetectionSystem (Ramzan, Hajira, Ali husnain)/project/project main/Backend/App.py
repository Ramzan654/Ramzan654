from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
import googleapiclient.discovery
import googleapiclient.errors
import transformers
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import urllib.parse
import uuid
from threading import Thread
import time

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load trained XLM-RoBERTa model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-model")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-model")

# Initialize YouTube API client
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyBED2f_e0YJs_YzpmwrNbeKnFBzF1CN838" # Replace with your actual YouTube Data API key
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Progress and results tracker
progress_tracker = {}
results_tracker = {}

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|@\S+|\S+@\S+', '', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    text = re.sub(r'\d+', 'num', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Extract video ID
def extract_video_id(url):
    query = urllib.parse.urlparse(url)
    if query.hostname in ('youtu.be', 'www.youtu.be'):
        return query.path[1:]
    if query.hostname in ('youtube.com', 'www.youtube.com'):
        if query.path == '/watch':
            p = urllib.parse.parse_qs(query.query)
            return p.get('v', [None])[0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

# Get channel name
def get_channel_name(video_id):
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if response['items']:
            return response['items'][0]['snippet']['channelTitle']
        return "Unknown Channel"
    except Exception as e:
        print(f"Error getting channel name: {e}")
        return "Unknown Channel"

# Fetch comments
def get_youtube_comments(video_id, max_comments=1000):
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except googleapiclient.errors.HttpError as e:
            print(f"HTTP error: {e}")
            break
    return comments

# Prediction
def predict_hate_speech(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
        # Assuming index 1 is 'hate' and index 0 is 'non-hate'
        hate_score = predictions[0][1].item()
        return hate_score
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.0

# IMPROVED Async task with smoother progress updates
def run_analysis_task(task_id, data):
    try:
        url = data.get("url")
        video_id = extract_video_id(url)
        
        if not video_id:
            progress_tracker[task_id] = {"progress": -1, "error": "Invalid YouTube URL"}
            return
        
        # Initial progress update
        progress_tracker[task_id] = {"progress": 5, "total": 0, "status": "Fetching video info..."}
        
        # Get channel name
        channel_name = get_channel_name(video_id)
        progress_tracker[task_id] = {"progress": 10, "total": 0, "status": "Fetching comments..."}
        
        # Fetch comments
        comments = get_youtube_comments(video_id)
        if not comments:
            progress_tracker[task_id] = {"progress": -1, "error": "No comments found or unable to fetch comments"}
            return
            
        total = len(comments)
        progress_tracker[task_id] = {"progress": 15, "total": total, "status": f"Processing {total} comments..."}
        
        hate_count = 0
        analyzed_comments_with_labels = [] # List to store comments with their labels
        batch_size = 5  # Process in smaller batches for smoother progress
        
        for i, comment in enumerate(comments):
            try:
                processed = preprocess(comment)
                score = predict_hate_speech(processed)
                label = "Hate" if score > 0.5 else "Non-Hate"
                
                if label == "Hate":
                    hate_count += 1
                
                analyzed_comments_with_labels.append({"comment": comment, "label": label}) # Store comment and label
                
                # Update progress more frequently (every comment or every batch)
                if i % batch_size == 0 or i == total - 1:
                    # Calculate progress: 15% (setup) + 80% (processing) + 5% (finalizing)
                    processing_progress = (i / total) * 80
                    current_progress = min(95, 15 + processing_progress)
                    
                    progress_tracker[task_id] = {
                        "progress": int(current_progress), 
                        "total": total,
                        "status": f"Analyzing comment {i+1}/{total}..."
                    }
                    
                    # Small delay for smooth updates
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error processing comment {i}: {e}")
                continue
        
        # Calculate final results
        hate_percentage = (hate_count / total) * 100 if total > 0 else 0
        non_hate_percentage = 100 - hate_percentage
        
        # Store results
        results_tracker[task_id] = {
            "hate_count": hate_count,
            "total": total,
            "hate_percentage": hate_percentage,
            "non_hate_percentage": non_hate_percentage,
            "channel_name": channel_name,
            "analyzed_comments": analyzed_comments_with_labels # Add the list of comments with labels
        }
        
        # Final progress update
        progress_tracker[task_id] = {
            "progress": 100, 
            "total": total,
            "status": "Analysis complete!"
        }
        
        print(f"Task {task_id} completed: {hate_count}/{total} hate comments ({hate_percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error in analysis task {task_id}: {e}")
        progress_tracker[task_id] = {"progress": -1, "error": str(e)}

# API endpoints
@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    try:
        data = request.json
        if not data or not data.get('url'):
            return jsonify({"error": "YouTube URL is required"}), 400
            
        task_id = str(uuid.uuid4())
        progress_tracker[task_id] = {"progress": 0, "total": 0, "status": "Starting analysis..."}
        
        # Start the analysis in a separate thread
        Thread(target=run_analysis_task, args=(task_id, data), daemon=True).start()
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return jsonify({"error": "Failed to start analysis"}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    try:
        progress_data = progress_tracker.get(task_id)
        if progress_data is None:
            return jsonify({"error": "Invalid task ID"}), 404
        return jsonify(progress_data), 200
    except Exception as e:
        print(f"Error getting progress for {task_id}: {e}")
        return jsonify({"error": "Failed to get progress"}), 500

@app.route('/results/<task_id>')
def get_results(task_id):
    try:
        results = results_tracker.get(task_id)
        if results is None:
            return jsonify({"error": "Results not available"}), 404
        return jsonify(results), 200
    except Exception as e:
        print(f"Error getting results for {task_id}: {e}")
        return jsonify({"error": "Failed to get results"}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"}), 200


if __name__ == '__main__':
    print("Starting Hate Speech Detection Server...")
    print("Model loaded successfully")
    # For local development, set host to '0.0.0.0' to be accessible from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5000)

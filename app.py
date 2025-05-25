from flask import Flask, render_template, request, redirect, url_for,session
from audio_bias_logic import calculate_score_from_form
from image_bias_logic import run_image_bias_pipeline, run_image_bias_pipeline_from_csv
from video_bias_logic import compute_video_bias_scores_from_upload
from text_bias_logic import run_text_bias_pipeline, compute_score_from_features
# from flask import redirect, url_for, session, flash
import os
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        input_values = {key: float(request.form.get(key, 0)) for key in request.form}
        results = calculate_score_from_form(input_values)

        # Store results and input temporarily in session
        session['results'] = results
        session['input_values'] = input_values

        # Redirect to avoid form resubmission on refresh
        return redirect(url_for('audio'))

    # On GET, retrieve from session then clear
    results = session.pop('results', None)
    input_values = session.pop('input_values', {})

    return render_template('audio.html', results=results, input_values=input_values)

    # return render_template('audio.html', results=results, input_values=input_values)

@app.route('/video', methods=['GET'])
def video():
    return render_template('video.html', results=None)

@app.route('/video/upload', methods=['POST'])
def video_upload():
    frame_file = request.files.get("frame_file")
    embed_file = request.files.get("embed_file")
    gender_file = request.files.get("gender_file")

    if not frame_file or not embed_file or not gender_file:
        return "All three files are required.", 400

    # Save uploaded files
    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_file.filename)
    embed_path = os.path.join(app.config['UPLOAD_FOLDER'], embed_file.filename)
    gender_path = os.path.join(app.config['UPLOAD_FOLDER'], gender_file.filename)

    frame_file.save(frame_path)
    embed_file.save(embed_path)
    gender_file.save(gender_path)

    # Run logic
    results = compute_video_bias_scores_from_upload(frame_path, embed_path, gender_path)
    return render_template("video.html", results=results)

@app.route('/text', methods=['GET', 'POST'])
def text():
    results = None
    if request.method == 'POST':
        file = request.files.get('file')
        mode = request.form.get('mode')  # Get selected mode from form

        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                if mode == 'features_only':
                    output_df = compute_score_from_features(file_path)
                else:
                    output_df = run_text_bias_pipeline(file_path)

                results = output_df.to_dict(orient='records')

            except Exception as e:
                results = f"Error during analysis: {e}"

    return render_template('text.html', results=results)

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/image/upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        csv_file = request.files['csv_file']
        if csv_file:
            file_path = f"temp_{csv_file.filename}"
            csv_file.save(file_path)

            # ✅ CSV VALIDATION STEP
            df = pd.read_csv(file_path)
            required_cols = ["Object_Class", "Gender", "Relative_Size", "Inverted_3D_Distance", "Inverted_Normalized_Depth", "Scene_Similarity_Bias"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return f"❌ Missing columns in CSV: {', '.join(missing_cols)}", 400

            # ✅ Continue with pipeline
            results_df = run_image_bias_pipeline_from_csv(file_path)
            top_male = results_df[results_df["Bias_Category"] == "Male-Biased"].head(5)
            top_female = results_df[results_df["Bias_Category"] == "Female-Biased"].head(5)
            top_neutral = results_df[results_df["Bias_Category"] == "Neutral"].head(5)
            return render_template("image_analysis.html",
                                   top_male=top_male.to_dict(orient='records'),
                                   top_female=top_female.to_dict(orient='records'),
                                   top_neutral=top_neutral.to_dict(orient='records'),
                                   results=results_df.to_dict(orient='records'))

    return render_template("image_upload.html")



@app.route('/image/analyze', methods=['POST'])
def analyze_image_bias():
    if 'csv_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['csv_file']
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    df = run_image_bias_pipeline_from_csv(filepath)
    
    # Separate top 5 objects per category
    top_male = df[df["Bias_Category"] == "Male-Biased"].nlargest(5, "Bias_Score").to_dict(orient="records")
    top_female = df[df["Bias_Category"] == "Female-Biased"].nsmallest(5, "Bias_Score").to_dict(orient="records")
    top_neutral = df[df["Bias_Category"] == "Neutral"].head(5).to_dict(orient="records")
    full_results = df.to_dict(orient="records")

    return render_template("image_analysis.html", top_male=top_male, top_female=top_female, top_neutral=top_neutral, results=full_results)

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.secret_key = 'resai_demo_tool_2025'
    app.run(debug=True)
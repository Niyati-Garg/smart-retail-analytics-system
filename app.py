import os
from flask import Flask, request, redirect, render_template, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from analysis import run_customer_analysis_multi
from reid_matcher import run_reid_merge
from session_merger import (
    get_session_folders_by_range,
    combine_csvs,
    combine_heatmaps,
    save_combined_heatmaps,
    combine_summary
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No video files uploaded", 400

    files = request.files.getlist('video')
    saved_paths = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            saved_paths.append(save_path)

    if not saved_paths:
        return "No valid video files uploaded.", 400

    timestamp = datetime.now()
    session_date = timestamp.strftime('%Y%m%d')
    output_folder = f'static/session_{session_date}'
    os.makedirs(output_folder, exist_ok=True)

    print("📁 Upload received. Saved videos:", saved_paths)
    print("📂 Output folder:", output_folder)

    run_customer_analysis_multi(saved_paths, output_folder)

    with open('latest_session.txt', 'w') as f:
        f.write(output_folder)

    return redirect('/')

@app.route('/latest_session.txt')
def latest():
    if not os.path.exists('latest_session.txt'):
        return "No session available", 404
    return send_file('latest_session.txt', mimetype='text/plain')

@app.route('/api/summary')
def summary_api():
    range_type = request.args.get('range', 'today')

    if range_type == 'today':
        if not os.path.exists('latest_session.txt'):
            return 'No session yet', 404
        with open('latest_session.txt') as f:
            folder = f.read().strip()
    else:
        today = datetime.now()
        if range_type == 'week':
            folders = get_session_folders_by_range('week', today.strftime('%Y'), today.strftime('%U'))
            output_folder = 'static/weekly_combined'
        elif range_type == 'month':
            folders = get_session_folders_by_range('month', today.strftime('%Y'), today.strftime('%m'))
            output_folder = 'static/monthly_combined'
        else:
            return 'Invalid range', 400

        if not folders:
            return 'No sessions found for this range.', 404

        df = combine_csvs(folders)
        zone, motion = combine_heatmaps(folders)
        save_combined_heatmaps(zone, motion, output_folder)
        combine_summary(df, output_folder)

        folder = output_folder

    return folder

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    app.run(debug=True, use_reloader=False)

# YouTube Comment Downloader

**A simple Python script to download comments from YouTube videos.**

---

## 🚀 Features
- Fetches comments using the `youtube-comment-downloader` package
- Saves output to CSV or JSON
- Easy to extend for sentiment analysis or other post-processing

---

## 📋 Prerequisites
- Python 3.8+
- `youtube-comment-downloader`
- `pandas`
- `requests`

You can install them with:

```bash
pip install youtube-comment-downloader pandas requests
```

---

## 🛠️ Installation & Usage

1. Clone the repo:  
   ```bash
   git clone https://github.com/YOUR_USERNAME/REPO.git
   cd REPO
   ```
2. (Optional) Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:  
   ```bash
   python pre-post.py --video_id <YOUTUBE_VIDEO_ID> --output comments.csv
   ```

---

## ⚙️ Configuration

| Option        | Description                             | Default      |
| ------------- | --------------------------------------- | ------------ |
| `--video_id`  | ID of the YouTube video to fetch        | *required*   |
| `--output`    | Path to write comments (CSV or JSON)    | `comments.csv` |

---

## 📂 Project Structure
```
.
├── pre-post.py          # Main downloader logic
├── requirements.txt     # Pinned dependencies
├── README.md            # This file
└── .gitignore           # Files/folders to ignore
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ✍️ Author
Esha Pandey – [Your GitHub profile link]

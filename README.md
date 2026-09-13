<h1 align="center">WATCHDOG</h1>

<p align="center">
<i>"A simple, AI-powered local media manager — find people in your photos and videos."</i>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python Badge">
<img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" alt="Streamlit Badge">
<img src="https://img.shields.io/badge/dlib-Face_Recognition-blue" alt="dlib Badge">
<img src="https://img.shields.io/badge/OpenCV-Computer_Vision-orange%3Flogo%3Dopencv" alt="OpenCV Badge">
<img src="https://img.shields.io/badge/SQLite-Database-blue%3Flogo%3Dsqlite" alt="SQLite Badge">
<img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License Badge">
</p>

---

## About The Project

WATCHDOG is a lightweight AI-powered facial recognition app written entirely in Python.
It was built as a portfolio project to demonstrate modern AI/ML integration, database management, and app development.

*It's not Google Photos — but it recognizes faces like it.*

**WATCHDOG** operates on the principle of facial recognition. Instead of searching by filenames or text, it searches by who is in your media.
It includes two primary systems within one app:

- **The Ingester**: A tab in the app that scans your uploaded videos or photos. It uses dlib and face clustering (DBSCAN) to automatically find and group unique people, letting you label them just once.
- **The Search App**: A Streamlit UI that lets you search your media library. You can scan your `images_to_search` folder, upload photos directly, or scan your `videos_to_search` folder to get timestamped reports of who was found.

---

## Features

**WATCHDOG** uses a local AI stack to find people in your local media:

- **Facial recognition**: find people across your entire photo and video library.
- **Smart ingestion**: upload videos to auto-cluster new, unknown faces, or upload photos to label every face one-by-one.
- **Local-first database**: all face encodings (biometric data) are stored 100% locally in a `face_database.db` (SQLite) file. Your data never leaves your computer.
- **Selective search**: a multi-select dropdown lets you choose exactly who you want to search for.
- **Multi-format search**: scan your `images_to_search` folder, upload photos directly for in-memory search, or scan your `videos_to_search` folder for timestamped reports.
- **Simple web UI**: a clean, multi-page web app built with Streamlit to handle all operations.

---

## Getting Started

### Prerequisites

```bash
Python 3.x
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Installing `dlib` can be difficult — make sure you have `cmake` and C++ build tools installed first.

### Installation

Clone this repository and enter the project directory:

```bash
git clone https://github.com/D-Majumder/watchDOG.git
cd watchDOG
```

### Usage

Using WATCHDOG is a simple, all-in-one process:

1. (Optional) Add your personal images to the `images_to_search` folder and videos to the `videos_to_search` folder.
2. Launch the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
   A browser window will open automatically.
3. Navigate to the "Ingest & Label" tab to teach the app new faces from your videos or photos.
4. Navigate to the "Search Photos" and "Search Videos" tabs to find them.

---

## Project Philosophy

WATCHDOG was built to learn by building a modern, end-to-end AI application. It covers:

- How facial recognition and face encodings work.
- How to use face clustering (DBSCAN) to group unknown faces automatically.
- How to use a relational database (SQLite) to store and query biometric data.
- How to build a clean, multi-page web UI for a complex AI tool using Streamlit.

---

## Disclaimer

This project is for educational and portfolio purposes only. It is a demonstration of AI integration and is not intended as a replacement for professional media management software. Handle all biometric data responsibly.

---

## Built With

- Python
- Streamlit
- dlib & face_recognition
- OpenCV
- SQLite
- scikit-learn (DBSCAN)

---

## Author

<p align="center">
<a href="mailto:dhrubamajumder@proton.me" target="_blank">
<img src="https://img.shields.io/badge/Email-Dhruba%20Majumder-blue?logo=gmail" alt="Email Badge">
</a>
<a href="https://www.linkedin.com/in/iamdhrubamajumder/" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-Dhruba%20Majumder-blue?logo=linkedin" alt="LinkedIn Badge">
</a>
<a href="https://github.com/D-Majumder" target="_blank">
<img src="https://img.shields.io/badge/GitHub-D--Majumder-black?logo=github" alt="GitHub Badge">
</a>
</p>

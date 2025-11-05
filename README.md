<h1 align="center" id="title">👤 WATCHDOG 👤</h1>

<p align="center">
<i>“A simple, AI-powered local media manager — find people in your photos and videos.”</i>
</p>

<p align="center">
<img src="https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt6814c99ef8071ed9/682dffd211a651b72ba9d96d/illustration-search-ai-with-logo-white.png" alt="AI Search Visual" style="max-width:100%;height:auto;border-radius:12px;box-shadow:0 0 15px rgba(0,150,255,0.3);">
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

<div align="center">
<img src="https://img.shields.io/badge/👤_Find_People_in_Your_Media_Instantly_-blue?style=for-the-badge" alt="Find People">
</div>

## ⚙️ About The Project

WATCHDOG is a lightweight AI-powered facial recognition app written entirely in Python.
It was built as a portfolio project to demonstrate modern AI/ML integration, database management, and app development.

> 🧩 *It’s not Google Photos — but it recognizes faces like it.*

**WATCHDOG** operates on the principle of facial recognition. Instead of searching by filenames or text, it searches by who is in your media.
It includes two primary systems within one app:

- 🧠 **The Ingester**: A tab in the app that scans your uploaded videos or photos. It uses dlib and Face Clustering (DBSCAN) to automatically find and group unique people, letting you label them just once.  
- 🔍 **The Search App**: A Streamlit UI that lets you search your media library. You can scan your images_to_search folder, upload photos directly, or scan your videos_to_search folder to get timestamped reports of who was found.  

---

## 🚨 Features
**WATCHDOG** uses a modern AI stack to find people in your local media:

✅ **Facial Recognition**: Find people across your entire photo and video library.  
✅ **Smart Ingestion**: Upload videos to auto-cluster new, unknown faces, or upload photos to label every face one-by-one.  
✅ **Local-First Database**: All face encodings (biometric data) are stored 100% locally in a face_database.db (SQLite) file. Your data never leaves your computer.  
✅ **Selective Search**: A multi-select dropdown lets you choose exactly who you want to search for.  
✅ **Multi-Format Search**: Scan your images_to_search folder, upload photos directly for in-memory search, or scan your videos_to_search folder for timestamped reports.  
✅ **Simple Web UI**: A clean, multi-page web app built with Streamlit to handle all operations.  

---

## 🚀 Getting Started

🧩 Prerequisites

```bash
Python 3.x
All libraries from requirements.txt
Install all dependencies using:

pip install -r requirements.txt
```
(Note: Installing dlib can be difficult. Make sure you have cmake and C++ build tools installed first.)

---

## 💻 Installation

Clone this repository and enter the project directory:

```bash
git clone [https://github.com/D-Majumder/WATCHDOG.git](https://github.com/D-Majumder/WATCHDOG.git)
cd WATCHDOG
```
(Note: You may need to rename your repository on GitHub to WATCHDOG for this link to be perfect.)

---

## ⚡ Usage

Using WATCHDOG is a simple, all-in-one process:

```bash
(Optional) Add your personal images to the images_to_search folder and videos to the videos_to_search folder.
Launch the Streamlit web app:
streamlit run app.py
```
(Note: A browser window will open automatically.)


- Navigate to the "📥 Ingest & Label" tab to teach the app new faces from your videos or photos.

- Navigate to the "🖼️ Search Photos" and "🎬 Search Videos" tabs to find them!

---

## 🧩 Project Philosophy

- 🧠 Learn by building modern, end-to-end AI applications.

WATCHDOG helps beginners understand:

- How "Facial Recognition" and "Face Encodings" work.
- How to use Face Clustering (DBSCAN) to group unknown faces automatically.
- How to use a relational database (SQLite) to store and query biometric data.
- How to build a clean, multi-page web UI for a complex AI tool using Streamlit.

--- 

## 📜 Disclaimer

This project is for educational and portfolio purposes only. It is a demonstration of AI integration and is not intended as a replacement for professional media management software. Handle all biometric data responsibly.

---

## 🛠️ Built With

- Python 🐍
- Streamlit 🎈
- dlib & face_recognition 🤖
- OpenCV 📸
- SQLite 🗃️
- scikit-learn (DBSCAN) 📊

---

## 🤝 Connect With Me

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

<div align="center">
<img src="https://img.shields.io/badge/🚀_Built_for_Portfolio_-Learn_Face_Recognition-green?style=for-the-badge" alt="Portfolio Project Badge">
</div>
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=1E90FF&height=100&section=footer&text=Search+your,+gallery+offline.&fontSize=22&fontColor=111111&animation=fadeIn" /> </p>

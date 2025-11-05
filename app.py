import streamlit as st
import face_recognition
import cv2
import os
import pickle
import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
import math

# --- Configuration ---
DB_NAME = "face_database.db"
TEMP_DIR = "temp_uploads"
FACES_DIR = "known_faces"
SCAN_DIR_IMAGES = "images_to_search"
SCAN_DIR_VIDEOS = "videos_to_search"
OUTPUT_DIR_IMAGES = "output_matches_images"
TOLERANCE = 0.55
VIDEO_PROCESS_SKIP_FRAMES = 15
SIGHTING_COOLDOWN = 5

# --- Directory Setup ---
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(SCAN_DIR_IMAGES, exist_ok=True)
os.makedirs(SCAN_DIR_VIDEOS, exist_ok=True)
os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)


# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS encodings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        encoding BLOB NOT NULL,
        FOREIGN KEY (person_id) REFERENCES people (id)
    )
    ''')
    conn.commit()
    conn.close()

# --- Database Helper Functions ---

def get_all_people():
    """Fetches all known people from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM people ORDER BY name")
    people = c.fetchall()
    conn.close()
    return people

def get_encodings_for_person(person_id):
    """Fetches all encodings for a specific person."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT encoding FROM encodings WHERE person_id = ?", (person_id,))
    rows = c.fetchall()
    conn.close()
    return [pickle.loads(row[0]) for row in rows]

def get_all_known_data(person_ids=None):
    """
    Loads all encodings and names from the DB into memory.
    If person_ids is provided, only loads data for those people.
    """
    all_encodings = []
    all_names = []
    
    people = get_all_people()
    
    if person_ids is not None:
        people = [p for p in people if p[0] in person_ids]

    for person_id, person_name in people:
        encodings = get_encodings_for_person(person_id)
        for enc in encodings:
            all_encodings.append(enc)
            all_names.append(person_name)
    return all_encodings, all_names

def add_person(name):
    """Adds a new person to the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO people (name) VALUES (?)", (name,))
        person_id = c.lastrowid
        conn.commit()
        conn.close()
        return person_id
    except sqlite3.IntegrityError:
        st.error(f"A person named '{name}' already exists.")
        return None

def add_encoding(person_id, encoding):
    """Adds a new face encoding for a person."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    enc_blob = pickle.dumps(encoding)
    c.execute("INSERT INTO encodings (person_id, encoding) VALUES (?, ?)", (person_id, enc_blob))
    conn.commit()
    conn.close()

def update_person_name(person_id, new_name):
    """Renames a person in the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE people SET name = ? WHERE id = ?", (new_name, person_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error(f"Failed to rename: A person named '{new_name}' already exists.")
        return False

# --- Core Face Logic ---

def format_time(seconds):
    """Converts seconds into MM:SS format."""
    minutes = math.floor(seconds / 60)
    seconds = math.floor(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_photos_for_labeling(uploaded_photos):
    """
    Scans uploaded photos, finds all faces, and returns a list
    of faces (encodings and thumbnails) for one-by-one labeling.
    """
    faces_to_label = []
    
    for uploaded_file in uploaded_photos:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None: continue
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            face_thumbnail = rgb_image[top:bottom, left:right]
            faces_to_label.append({
                "encoding": enc,
                "thumbnail": face_thumbnail
            })
    return faces_to_label


def process_video_for_clustering(video_path, progress_bar):
    """
    Scans a video, finds all faces, clusters them, and returns
    a list of unique faces (encodings and thumbnails) for labeling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return []

    all_encodings = []
    all_face_thumbnails = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress_bar.progress(frame_count / total_frames, text=f"Scanning frame {frame_count}/{total_frames}...")

        if frame_count % VIDEO_PROCESS_SKIP_FRAMES != 0:
            continue

        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            continue

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            all_encodings.append(enc)
            
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            face_thumbnail = frame[top:bottom, left:right]
            all_face_thumbnails.append(face_thumbnail)

    cap.release()
    progress_bar.empty()

    if not all_encodings:
        st.warning("No faces found in the video.")
        return []

    st.write(f"Found {len(all_encodings)} total face instances. Now clustering...")

    clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=TOLERANCE)
    clt.fit(all_encodings)
    
    unique_labels = set(clt.labels_)
    st.write(f"Found {len(unique_labels) - (1 if -1 in unique_labels else 0)} unique people.")

    unique_people = []
    for label in unique_labels:
        if label == -1:
            continue # -1 is the "noise" label (faces that didn't fit a cluster)

        indices = np.where(clt.labels_ == label)[0]
        
        person_encodings = [all_encodings[i] for i in indices]
        person_thumbnails = [all_face_thumbnails[i] for i in indices]
        
        best_thumbnail = person_thumbnails[0]
        best_thumbnail = cv2.cvtColor(best_thumbnail, cv2.COLOR_BGR2RGB)
        
        unique_people.append({
            "id": int(label),
            "encodings": person_encodings,
            "thumbnail": best_thumbnail
        })
        
    return unique_people


def scan_video_for_people(video_path, known_encodings, known_names, tolerance, progress_bar):
    """
    Scans a single video and returns a report of *when* known
    people were seen.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        
    frame_count = 0
    findings = {}
    last_seen = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress_bar.progress(frame_count / total_frames, text=f"Scanning video... (Frame {frame_count}/{total_frames})")

        if frame_count % VIDEO_PROCESS_SKIP_FRAMES != 0:
            continue
            
        timestamp_sec = frame_count / fps

        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            continue

        face_locations = face_recognition.face_locations(rgb_small_frame)
        unknown_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for unknown_encoding in unknown_encodings:
            results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
            
            if True in results:
                match_index = results.index(True)
                name = known_names[match_index]
                
                if name not in last_seen or (timestamp_sec - last_seen[name] > SIGHTING_COOLDOWN):
                    if name not in findings:
                        findings[name] = []
                    
                    findings[name].append(timestamp_sec)
                    last_seen[name] = timestamp_sec

    cap.release()
    progress_bar.empty()
    return findings


# --- Streamlit UI Application ---

st.set_page_config(layout="wide", page_title="Face Finder Pro")
st.title("👤 Face Finder Pro")
st.write("An advanced media tool to index and find people in photos and videos.")

init_db()

def render_labeling_ui(face_queue_key, display_caption_prefix):
    """
    Renders the common UI for labeling faces from a session state queue.
    """
    all_people = get_all_people()
    people_options = {name: person_id for person_id, name in all_people}
    
    if st.session_state[face_queue_key]:
        face_to_label = st.session_state[face_queue_key][0]
        
        face_key = hash(face_to_label['encoding'].tobytes())
        
        st.image(face_to_label['thumbnail'], width=150, caption=f"{display_caption_prefix}")
        
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("New Name:", key=f"new_name_{face_key}")
            if st.button("Save as New Person", key=f"save_new_{face_key}"):
                if new_name.strip():
                    new_person_id = add_person(new_name)
                    if new_person_id:
                        # If from video, 'encodings' is a list. If from photo, 'encoding' is single.
                        encodings_to_add = face_to_label.get('encodings', [face_to_label['encoding']])
                        
                        for enc in encodings_to_add:
                            add_encoding(new_person_id, enc)
                        
                        thumb_path = os.path.join(FACES_DIR, f"{new_person_id}_{new_name}.jpg")
                        thumb_bgr = cv2.cvtColor(face_to_label['thumbnail'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(thumb_path, thumb_bgr)
                        
                        st.success(f"Saved '{new_name}' to database!")
                        st.session_state[face_queue_key].pop(0)
                        st.rerun()
                else:
                    st.warning("Please enter a name.")

        with col2:
            if people_options:
                selected_name = st.selectbox("Or Add to Existing:", 
                                             options=[""] + list(people_options.keys()), 
                                             key=f"select_{face_key}")
                if st.button("Add to Existing", key=f"save_existing_{face_key}"):
                    if selected_name:
                        person_id = people_options[selected_name]
                        encodings_to_add = face_to_label.get('encodings', [face_to_label['encoding']])
                        for enc in encodings_to_add:
                            add_encoding(person_id, enc)
                        st.success(f"Added new face samples to '{selected_name}'.")
                        st.session_state[face_queue_key].pop(0)
                        st.rerun()
                    else:
                        st.warning("Please select a person.")
        st.divider()


tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home & People", 
    "📥 Ingest & Label", 
    "🖼️ Search Photos", 
    "🎬 Search Videos"
])

with tab1:
    st.header("Known People")
    st.write("This is everyone the system has learned to recognize.")
    
    people = get_all_people()
    
    if not people:
        st.info("No people found in the database. Go to the 'Ingest & Label' tab to add people.")
    else:
        cols = st.columns(5) 
        col_index = 0
        
        for person_id, name in people:
            with cols[col_index % 5]:
                face_img_path = os.path.join(FACES_DIR, f"{person_id}_{name}.jpg")
                
                if os.path.exists(face_img_path):
                    st.image(face_img_path, width=150)
                else:
                    st.image("https://placehold.co/150x150/333/FFF?text=No\nImage", width=150)
                
                new_name = st.text_input(f"Name:", value=name, key=f"name_{person_id}")
                if new_name != name and new_name.strip():
                    if st.button("Rename", key=f"rename_{person_id}"):
                        if update_person_name(person_id, new_name):
                            new_img_path = os.path.join(FACES_DIR, f"{person_id}_{new_name}.jpg")
                            if os.path.exists(face_img_path):
                                os.rename(face_img_path, new_img_path)
                            st.success(f"Renamed '{name}' to '{new_name}'")
                            st.rerun()
            col_index += 1


with tab2:
    st.header("Ingest & Label New Faces")
    st.write("Upload media to teach the system new faces.")

    st.session_state.setdefault('video_faces_to_label', [])
    st.session_state.setdefault('photo_faces_to_label', [])

    st.subheader("Option 1: Ingest from Video (Good for finding new people)")
    st.write("This scans a video, groups (clusters) similar faces, and shows you each *unique* person to label.")
    
    uploaded_video = st.file_uploader("Upload a video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"], key="video_uploader")
    
    if uploaded_video is not None:
        st.session_state['photo_faces_to_label'] = []
        
        temp_path = os.path.join(TEMP_DIR, uploaded_video.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.write("Video uploaded. Starting face analysis...")
        st.info("This is a slow process, especially for long videos. Please wait.")
        progress_bar = st.progress(0, text="Starting scan...")
        
        unique_people = process_video_for_clustering(temp_path, progress_bar)
        
        os.remove(temp_path)
        
        if unique_people:
            st.success("Analysis complete!")
            st.write(f"Found {len(unique_people)} unique people. Please label them or add them to an existing person:")
            st.session_state['video_faces_to_label'] = unique_people
        
    if st.session_state['video_faces_to_label']:
        render_labeling_ui('video_faces_to_label', "Unknown Person (from Video)")

    st.divider()

    st.subheader("Option 2: Ingest from Photos (Good for adding samples)")
    st.write("This scans photos and shows you *every* face it finds, one by one, for you to label.")

    uploaded_photos = st.file_uploader("Upload photos (.png, .jpg, .jpeg)", 
                                       type=["png", "jpg", "jpeg"], 
                                       accept_multiple_files=True, 
                                       key="photo_uploader")
    
    if uploaded_photos:
        st.session_state['video_faces_to_label'] = []
        
        st.write(f"Processing {len(uploaded_photos)} photo(s)...")
        new_faces = process_photos_for_labeling(uploaded_photos)
        
        if new_faces:
            st.session_state['photo_faces_to_label'] = new_faces
            st.success(f"Found {len(new_faces)} faces to label.")
        else:
            st.warning("No faces were found in the uploaded photo(s).")
    
    if st.session_state['photo_faces_to_label']:
        render_labeling_ui('photo_faces_to_label', f"Face {len(st.session_state['photo_faces_to_label'])} (from Photo)")


with tab3:
    st.header("Search for People in Photos")
    
    people_list = get_all_people()
    if not people_list:
        st.error("No known people in the database. Please go to 'Ingest & Label' to add people first.")
    else:
        people_map = {name: person_id for person_id, name in people_list}
        selected_names = st.multiselect("Select people to search for:", 
                                        options=people_map.keys(), 
                                        default=list(people_map.keys()))
        
        if not selected_names:
            st.warning("Please select at least one person to search for.")
        else:
            selected_ids = [people_map[name] for name in selected_names]
            
            st.subheader("Search Uploaded Photos")
            st.write("Upload one or more photos to scan them immediately (in-memory).")
            
            uploaded_files = st.file_uploader("Upload photos (.png, .jpg, .jpeg)", 
                                              type=["png", "jpg", "jpeg"], 
                                              accept_multiple_files=True,
                                              key="search_photo_uploader")
            
            if uploaded_files:
                known_encodings, known_names = get_all_known_data(selected_ids)
                
                st.write(f"Scanning {len(uploaded_files)} uploaded image(s) for {selected_names}...")
                
                match_images = []
                no_match_images = []
                
                progress_bar = st.progress(0, text="Starting scan...")
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files), text=f"Scanning {uploaded_file.name}...")
                    
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    if image is None: continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_image)
                    unknown_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                    if not unknown_encodings:
                        no_match_images.append(uploaded_file)
                        continue

                    found_in_this_image = False
                    for unknown_encoding in unknown_encodings:
                        results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=TOLERANCE)
                        if True in results:
                            found_in_this_image = True
                            break
                    
                    if found_in_this_image:
                        match_images.append(uploaded_file)
                    else:
                        no_match_images.append(uploaded_file)
                
                progress_bar.empty()
                st.success("Upload scan complete.")
                
                if match_images:
                    st.subheader("✅ Matches Found In:")
                    cols = st.columns(5)
                    for i, img in enumerate(match_images):
                        cols[i % 5].image(img, caption=img.name, use_container_width=True)

                if no_match_images:
                    st.subheader("❌ No Matches Found In:")
                    cols = st.columns(5)
                    for i, img in enumerate(no_match_images):
                        cols[i % 5].image(img, caption=img.name, use_container_width=True)

            st.divider()

            st.subheader("Search Photo Folder (Batch Mode)")
            st.write(f"Add photos to the '{SCAN_DIR_IMAGES}' folder, then click Scan.")
            st.info(f"This option is good for large batches. Results are **copied (original, no boxes)** to the '{OUTPUT_DIR_IMAGES}' folder.")

            if st.button("Start Photo Folder Scan"):
                known_encodings, known_names = get_all_known_data(selected_ids)
                
                st.write(f"Scanning '{SCAN_DIR_IMAGES}' for {selected_names}...")
                
                scanned_files = 0
                match_files = 0
                
                for f in os.listdir(OUTPUT_DIR_IMAGES):
                    if os.path.isfile(os.path.join(OUTPUT_DIR_IMAGES, f)):
                        os.remove(os.path.join(OUTPUT_DIR_IMAGES, f))

                progress_bar = st.progress(0, text="Starting photo scan...")
                image_files = [f for f in os.listdir(SCAN_DIR_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                total_files = len(image_files)
                
                if total_files == 0:
                    st.warning(f"No images found in '{SCAN_DIR_IMAGES}'.")
                else:
                    for i, filename in enumerate(image_files):
                        progress_bar.progress((i + 1) / total_files, text=f"Scanning {filename}...")
                        image_path = os.path.join(SCAN_DIR_IMAGES, filename)
                        
                        try:
                            image = cv2.imread(image_path)
                            if image is None: continue
                            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        except Exception:
                            continue
                        
                        scanned_files += 1
                        face_locations = face_recognition.face_locations(rgb_image)
                        unknown_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                        if not unknown_encodings:
                            continue

                        found_in_this_image = False
                        for unknown_encoding in unknown_encodings:
                            if not found_in_this_image:
                                results = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=TOLERANCE)
                                if True in results:
                                    found_in_this_image = True

                        if found_in_this_image:
                            output_path = os.path.join(OUTPUT_DIR_IMAGES, filename)
                            shutil.copy(image_path, output_path)
                            match_files += 1
                    
                    progress_bar.empty()
                    st.success(f"Scan complete! Scanned {scanned_files} images. Found matches in {match_files} images.")
                    st.write(f"Original images of matches are saved in '{OUTPUT_DIR_IMAGES}'.")
                
                st.header("✅ Results (Originals Copied to Output Folder)")
                result_images = [f for f in os.listdir(OUTPUT_DIR_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not result_images:
                    st.info("No matches found in any images.")
                else:
                    cols = st.columns(5)
                    for i, img_file in enumerate(result_images):
                        cols[i % 5].image(os.path.join(OUTPUT_DIR_IMAGES, img_file), caption=img_file, use_container_width=True)


with tab4:
    st.header("Search for People in Videos")
    
    people_list_video = get_all_people()
    if not people_list_video:
        st.error("No known people in the database. Please go to 'Ingest & Label' to add people first.")
    else:
        people_map_video = {name: person_id for person_id, name in people_list_video}
        selected_names_video = st.multiselect("Select people to search for:", 
                                              options=people_map_video.keys(), 
                                              default=list(people_map_video.keys()),
                                              key="video_multiselect")
        
        st.write(f"Add videos to the '{SCAN_DIR_VIDEOS}' folder, then click Scan.")
        st.info("Note: This is a *very* slow process. A 10-minute video can take a long time to scan.")
        
        if not selected_names_video:
            st.warning("Please select at least one person to search for.")
        else:
            if st.button("Start Video Scan"):
                selected_ids_video = [people_map_video[name] for name in selected_names_video]
                known_encodings, known_names = get_all_known_data(selected_ids_video)
                
                st.write(f"Scanning all videos in '{SCAN_DIR_VIDEOS}' for {selected_names_video}...")
                
                video_files = [f for f in os.listdir(SCAN_DIR_VIDEOS) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
                
                if not video_files:
                    st.warning(f"No videos found in '{SCAN_DIR_VIDEOS}'.")
                else:
                    for video_file in video_files:
                        st.subheader(f"Results for: {video_file}")
                        video_path = os.path.join(SCAN_DIR_VIDEOS, video_file)
                        
                        progress_bar = st.progress(0, text=f"Starting scan for {video_file}...")
                        
                        findings = scan_video_for_people(video_path, known_encodings, known_names, TOLERANCE, progress_bar)
                        
                        progress_bar.empty()
                        
                        if not findings:
                            st.write("No known people found in this video.")
                        else:
                            for name, timestamps in findings.items():
                                with st.expander(f"Found '{name}' at {len(timestamps)} sighting(s)"):
                                    for ts in timestamps:
                                        st.write(f"- Timestamp: {format_time(ts)}")
                        st.divider()
                    
                    st.success("All video scans complete.")
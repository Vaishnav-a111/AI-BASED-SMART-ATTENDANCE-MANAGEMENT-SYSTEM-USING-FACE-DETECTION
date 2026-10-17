

from flask import Flask,render_template,request,redirect,session,url_for
import mysql.connector
import cv2
import face_recognition
import os
import datetime
import pickle
import tempfile

#-------- LOAD ENV VARIABLES --------
def _load_env_manually():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Run manual loader as fallback/override to ensure .env is parsed
_load_env_manually()
import smtplib
import ssl
import re
import glob
from email.message import EmailMessage
from flask import send_from_directory
from twilio.rest import Client

# -------- HOLIDAY CHECK FUNCTION --------
def _get_admin_holiday_dates():
    """Return { 'YYYY-MM-DD': reason } for all admin-declared holidays."""
    cursor.execute("SELECT holiday_date, reason FROM admin_holidays")
    return {str(row[0]): (row[1] or "Holiday") for row in cursor.fetchall()}

def is_holiday(date_val=None):
    if date_val is None:
        date_val = datetime.date.today()

    # Sunday
    if date_val.weekday() == 6:
        return True

    # 2nd Saturday (8 to 14) and 4th Saturday (22 to 28)
    if date_val.weekday() == 5 and (8 <= date_val.day <= 14 or 22 <= date_val.day <= 28):
        return True

    cursor.execute(
        "SELECT 1 FROM admin_holidays WHERE holiday_date=%s LIMIT 1",
        (date_val,),
    )
    if cursor.fetchone():
        return True

    return False
def _norm_student_id(v):
    """Same student must match in marked set and DB rows (MySQL may return int/str/bytes)."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace").strip() or None
    return str(v).strip() or None

def _normalize_batch_value(b):
    if b is None:
        return ""
    s = str(b).strip()
    if not s or s.lower() == "none":
        return ""

    digits = re.findall(r"\d+", s)
    if digits:
        return f"B{digits[0]}"

    return "".join(c for c in s.upper() if c.isalnum())
app = Flask(__name__)
app.secret_key="attendance_secret"

_DB_CONFIG = dict(
host="localhost",
user="root",
password="Vaishnav8301946204",
database="Attendance",
connection_timeout=30,
)
_db_conn = None
_db_cursor = None

def get_db():
    """Return a live (cursor, connection) pair, reconnecting automatically if needed."""
    global _db_conn, _db_cursor
    try:
        if _db_conn is None or not _db_conn.is_connected():
            raise Exception("not connected")
        _db_conn.ping(reconnect=True, attempts=3, delay=1)
    except Exception:
        _db_conn = mysql.connector.connect(**_DB_CONFIG)
        _db_cursor = _db_conn.cursor()
    return _db_cursor, _db_conn

# Bootstrap a connection immediately so the migration below works
cursor, db = get_db()

@app.before_request
def _ensure_db():
    """Reconnect to MySQL before every request so the connection never goes stale."""
    global cursor, db
    cursor, db = get_db()

# Check and add 'month' column to timetable if it doesn't exist
try:
    cursor.execute("SHOW COLUMNS FROM timetable LIKE 'month'")
    if not cursor.fetchone():
        cursor.execute("ALTER TABLE timetable ADD COLUMN month VARCHAR(20) NULL")
        db.commit()
        print("Successfully added 'month' column to 'timetable' table.")
except Exception as e:
    print("Database migration error (adding 'month'):", e)

# Check and add 'parent_phone' column to students if it doesn't exist
try:
    cursor.execute("SHOW COLUMNS FROM students LIKE 'parent_phone'")
    if not cursor.fetchone():
        cursor.execute("ALTER TABLE students ADD COLUMN parent_phone VARCHAR(20) NULL")
        db.commit()
        print("Successfully added 'parent_phone' column to 'students' table.")
except Exception as e:
    print("Database migration error (adding 'parent_phone'):", e)

# Check and add 'profile_edit_status' column to students if it doesn't exist
try:
    cursor.execute("SHOW COLUMNS FROM students LIKE 'profile_edit_status'")
    if not cursor.fetchone():
        cursor.execute("ALTER TABLE students ADD COLUMN profile_edit_status VARCHAR(20) DEFAULT 'none'")
        db.commit()
        print("Successfully added 'profile_edit_status' column to 'students' table.")
except Exception as e:
    print("Database migration error (adding 'profile_edit_status'):", e)

# Check and add index idx_attendance_student to attendance if it doesn't exist
try:
    cursor.execute("SHOW INDEX FROM attendance WHERE Key_name = 'idx_attendance_student'")
    if not cursor.fetchone():
        cursor.execute("CREATE INDEX idx_attendance_student ON attendance (student_id)")
        db.commit()
        print("Successfully created index idx_attendance_student.")
except Exception as e:
    print("Database migration error (creating idx_attendance_student):", e)

# Check and add index idx_attendance_date to attendance if it doesn't exist
try:
    cursor.execute("SHOW INDEX FROM attendance WHERE Key_name = 'idx_attendance_date'")
    if not cursor.fetchone():
        cursor.execute("CREATE INDEX idx_attendance_date ON attendance (date)")
        db.commit()
        print("Successfully created index idx_attendance_date.")
except Exception as e:
    print("Database migration error (creating idx_attendance_date):", e)

# Admin-declared holidays (no classes for that date)
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_holidays (
            holiday_date DATE PRIMARY KEY,
            reason VARCHAR(255) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
except Exception as e:
    print("Database migration error (admin_holidays):", e)

# OD Requests table
try:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS od_requests (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(50),
            date DATE,
            subject VARCHAR(100),
            hour VARCHAR(50),
            reason TEXT,
            status VARCHAR(20) DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
except Exception as e:
    print("Database migration error (od_requests):", e)

# -------- GLOBAL FACE ENCODING CACHE --------
# Built ONCE at startup so take_attendance never re-reads images from disk.
# Call refresh_face_encoding_cache() whenever a profile photo changes.
_face_cache_lock = None # lazy-init (threading not imported yet at module level)
_known_encodings_cache = [] # list of 128-d numpy arrays
_known_norm_ids_cache = [] # parallel list of normalised student IDs
_id_to_batch_cache = {}
_id_to_raw_id_cache = {}

# Tolerance values are now loaded from config.
# It uses 68-point math, so false positives at a distance are completely eliminated.
from config import FACE_MATCH_TOLERANCE, FACE_MATCH_MARGIN, LIVE_ENCODING_JITTERS

# Margin loaded from config (FACE_MATCH_MARGIN imported from config)
MATCH_CONFIRM_FRAMES = 4  # Frames needed to confirm a match (lower = faster marking)

# Keep showing last recognized roll for this many bad frames (stops label flicker).
OVERLAY_STICKY_FRAMES = 10
FACE_ENCODING_JITTERS = 5
FACE_ENCODING_MODEL = "large"  # 68-point model for perfect accuracy from a distance
LIVE_ENCODING_JITTERS = LIVE_ENCODING_JITTERS  # from config

def _enhance_rgb_for_recognition(rgb_image):
    """Reduce glare / uneven light and normalize brightness for robust recognition.
    Handles: glasses reflections, side-angles, varied lighting, spectacle glare.
    """
    try:
        # Convert to LAB colour space for perceptual luminance processing
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Stronger CLAHE: clipLimit=3.0 reduces glare hotspots from glasses lenses
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        # Gamma correction to normalize very bright or very dark webcam frames
        # gamma=0.85 pulls down harsh highlights (spectacle reflections)
        import numpy as np
        inv_gamma = 1.0 / 0.85
        table = (np.arange(256) / 255.0) ** inv_gamma * 255
        lut = np.clip(table, 0, 255).astype(np.uint8)
        return lut[enhanced]
    except Exception:
        return rgb_image

def _student_reference_image_paths(nid, image_file, dataset_path="dataset"):
    """All dataset images for one student (main photo + extras for with/without glasses).
    Supports patterns like: 100.jpg, 100.jpeg, 100.png, 100_extra.jpg, 100_noglasses.jpg,
    100_glasses.jpg, 100_1.jpg, etc.
    """
    paths = []
    base = os.path.splitext(image_file or nid)[0]
    seen = set()
    # Primary candidates
    for cand in (
        image_file,
        f"{nid}.jpg",
        f"{nid}.jpeg",
        f"{nid}.png",
        f"{base}.jpg",
        f"{base}.jpeg",
        f"{base}.png",
    ):
        if not cand:
            continue
        fp = os.path.join(dataset_path, cand)
        if fp not in seen and os.path.isfile(fp):
            seen.add(fp)
            paths.append(fp)
    # Extra variants: any file starting with the student ID followed by _ or -
    # Covers: 100_extra.jpg, 100_noglasses.jpg, 100_glasses.jpg, 100-1.jpg, 100_2.png, etc.
    for pattern in (f"{nid}_*", f"{nid}-*"):
        for extra in glob.glob(os.path.join(dataset_path, pattern)):
            if extra not in seen and os.path.isfile(extra):
                seen.add(extra)
                paths.append(extra)
    return paths

def _encode_faces_from_image_file(filepath):
    """Return list of face encodings from one image (multiple jitters = robust to glasses/angle)."""
    try:
        img = face_recognition.load_image_file(filepath)
        img = _enhance_rgb_for_recognition(img)
        encodings = face_recognition.face_encodings(
            img,
            num_jitters=FACE_ENCODING_JITTERS,
            model=FACE_ENCODING_MODEL,
        )
        if encodings:
            return encodings
        # Retry with slight upscale if face is small in the reference photo
        h, w = img.shape[:2]
        if max(h, w) < 400:
            scaled = cv2.resize(img, (w * 2, h * 2))
            return face_recognition.face_encodings(
                scaled,
                num_jitters=FACE_ENCODING_JITTERS,
                model=FACE_ENCODING_MODEL,
            )
    except Exception as e:
        print(f"Encode error ({filepath}):", e)
    return []

def _uploaded_photo_matches_student(student_id, uploaded_file_path):
    """
    Detect the face in the newly uploaded file and compare it against
    the student's existing profile photo face encodings.
    Returns (True, None) if matching, (False, "error message") otherwise.
    """
    # 1. Get face encodings from the new photo
    new_encodings = _encode_faces_from_image_file(uploaded_file_path)
    if not new_encodings:
        return False, "No face detected in the uploaded photo. Please try a clearer picture with good lighting."

    # 2. Get existing student encodings
    student_refs = []
    norm_sid = _norm_student_id(student_id)

    for enc, nid in zip(_known_encodings_cache, _known_norm_ids_cache):
        if nid == norm_sid:
            student_refs.append(enc)
            
    # If no references exist in cache, fall back to calculating from disk files
    if not student_refs:
        try:
            cursor.execute("SELECT image FROM students WHERE id=%s", (student_id,))
            row = cursor.fetchone()
            image_file = row[0] if row else None
            for fp in _student_reference_image_paths(norm_sid, image_file, "dataset"):
                for enc in _encode_faces_from_image_file(fp):
                    student_refs.append(enc)
        except Exception as e:
            print("DB/Disk check failed in _uploaded_photo_matches_student:", e)
            
    if not student_refs:
        # No old photos to match against (could be a newly created student without a photo previously)
        return True, None
        
    # 3. Compare new face with existing references
    for new_enc in new_encodings:
        distances = face_recognition.face_distance(student_refs, new_enc)
        if len(distances) > 0 and float(distances.min()) <= FACE_MATCH_TOLERANCE:
            return True, None
            
    return False, "The uploaded photo does not match your existing profile photo. Please upload a valid photo of yourself."

def _find_existing_student_by_photo(uploaded_file_path):
    """
    Compare an uploaded photo against all enrolled students.
    Returns (student_id, student_name) if the face already belongs to someone, else None.
    """
    new_encodings = _encode_faces_from_image_file(uploaded_file_path)
    if not new_encodings:
        return None

    if not _known_encodings_cache:
        try:
            refresh_face_encoding_cache()
        except Exception:
            pass

    student_map = _student_encoding_map(_known_encodings_cache, _known_norm_ids_cache)
    if not student_map:
        return None

    for new_enc in new_encodings:
        match_id, _, _ = _match_face_to_student(new_enc, student_map)
        if not match_id:
            continue
        raw_id = _id_to_raw_id_cache.get(match_id, match_id)
        cursor.execute("SELECT name FROM students WHERE id=%s", (raw_id,))
        row = cursor.fetchone()
        return (raw_id, row[0] if row else match_id)
    return None

def _batch_encodings_from_cache(batch_norm):
    """Encodings for students in this batch only (from global cache)."""
    target = _normalize_batch_value(batch_norm)
    encodings, norm_ids = [], []
    for enc, nid in zip(_known_encodings_cache, _known_norm_ids_cache):
        if _id_to_batch_cache.get(nid) == target:
            encodings.append(enc)
            norm_ids.append(nid)
    return encodings, norm_ids

def _student_encoding_map(encodings, norm_ids):
    """Group reference encodings by roll number (one student may have several photos)."""
    grouped = {}
    for enc, nid in zip(encodings, norm_ids):
        grouped.setdefault(nid, []).append(enc)
    return grouped

def _match_face_to_student(live_encoding, student_map, tolerance=FACE_MATCH_TOLERANCE, margin=FACE_MATCH_MARGIN):
    """
    Pick one student when clearly closest, using AVERAGE distance across all
    reference encodings per student.  This is far more robust than min-distance
    because a single outlier encoding cannot pull the match to the wrong student.
    Returns (matched_roll_or_None, closest_roll, closest_distance).
    """
    if not student_map:
        return None, None, None

    import numpy as np
    per_student = []
    for nid, refs in student_map.items():
        distances = face_recognition.face_distance(refs, live_encoding)
        # Use MEAN distance — robust against outlier reference photos
        avg_dist = float(np.mean(distances))
        per_student.append((avg_dist, nid))

    per_student.sort(key=lambda item: item[0])
    best_dist, best_id = per_student[0]

    under_tol = [(d, nid) for d, nid in per_student if d <= tolerance]
    if not under_tol:
        return None, best_id, best_dist

    match_dist, match_id = under_tol[0]
    # Check margin against the global second best, even if the second best is over tolerance.
    # This completely eliminates false positives caused by a loose tolerance threshold.
    if len(per_student) > 1:
        second_dist = per_student[1][0]
        if (second_dist - match_dist) < margin:
            return None, best_id, best_dist

    return match_id, best_id, match_dist
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_encodings_cache.pkl")

def _load_pickle_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("Error loading face encoding cache file:", e)
    return {}

def invalidate_face_cache(student_id: str):
    """Invalidate face-encoding cache for the given student ID.
    Removes all dataset entries for that student from the pickle cache so the
    next refresh always re-encodes their updated photo from disk.
    """
    try:
        nid = _norm_student_id(student_id)
        if nid and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
                # Remove any entry whose filename contains the student's ID
                stale = [fp for fp in list(cache.keys())
                         if os.path.splitext(os.path.basename(fp))[0].startswith(nid)]
                for fp in stale:
                    cache.pop(fp, None)
                    print(f"[Face Cache] Invalidated entry: {fp}")
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[Face Cache] Pickle invalidation failed, will force full rebuild: {e}")
                try:
                    os.remove(CACHE_FILE)
                except Exception:
                    pass
    except Exception as exc:
        print('Cache invalidation error:', exc)

def _save_pickle_cache(cache):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print("Error saving face encoding cache file:", e)

def refresh_face_encoding_cache():
    """Load / reload every student photo from disk and cache the encodings.
    Safe to call from any thread; rebuilds atomically."""
    global _known_encodings_cache, _known_norm_ids_cache
    global _id_to_batch_cache, _id_to_raw_id_cache

    enc_list  = []
    nid_list  = []
    batch_map = {}
    raw_map   = {}

    try:
        tmp_db = mysql.connector.connect(
            host="localhost", user="root",
            password="Vaishnav8301946204", database="Attendance"
        )
        tmp_cur = tmp_db.cursor()
        tmp_cur.execute("SELECT * FROM students")
        cols = [c[0] for c in tmp_cur.description]
        rows = [dict(zip(cols, r)) for r in tmp_cur.fetchall()]
        tmp_cur.close()
        tmp_db.close()
    except Exception as e:
        print("Face cache DB error:", e)
        return

    # Load persistent cache
    cache = _load_pickle_cache()
    cache_updated = False
    processed_paths = set()

    dataset_path = "dataset"
    for row in rows:
        nid = _norm_student_id(row.get("id"))
        image_file = row.get("image") or row.get("filename")
        if not nid:
            continue

        batch_map[nid] = _normalize_batch_value(row.get("batch"))
        raw_map[nid] = row.get("id")

        for fp in _student_reference_image_paths(nid, image_file, dataset_path):
            try:
                mtime = os.path.getmtime(fp)
                size = os.path.getsize(fp)
                processed_paths.add(fp)
            except Exception:
                continue

            # Check if cache hit exists and is valid
            cached_data = cache.get(fp)
            if cached_data and cached_data.get("mtime") == mtime and cached_data.get("size") == size:
                encs = cached_data.get("encodings", [])
            else:
                print(f"[Face Cache] Recalculating face encodings for: {fp}")
                encs = _encode_faces_from_image_file(fp)
                cache[fp] = {
                    "mtime": mtime,
                    "size": size,
                    "encodings": encs
                }
                cache_updated = True

            for enc in encs:
                enc_list.append(enc)
                nid_list.append(nid)

    # Prune stale cache entries for deleted files
    stale_paths = [fp for fp in cache if fp not in processed_paths]
    if stale_paths:
        for fp in stale_paths:
            cache.pop(fp, None)
        cache_updated = True

    if cache_updated:
        _save_pickle_cache(cache)

    _known_encodings_cache  = enc_list
    _known_norm_ids_cache   = nid_list
    _id_to_batch_cache      = batch_map
    _id_to_raw_id_cache     = raw_map
    print(f"[Face Cache] {len(enc_list)} encoding(s) loaded for {len(set(nid_list))} student(s).")
# Build cache at startup
refresh_face_encoding_cache()

@app.route('/dataset/<filename>')
def dataset_file(filename):
    return send_from_directory('dataset', filename)

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")

# ---------------- ADMIN PANEL ----------------
@app.route('/admin_login',methods=['GET','POST'])
def admin_login():
    if request.method=="POST":
        username=request.form['username']
        password=request.form['password']

        if username=="admin" and password=="admin":
            session['admin']=True
            return redirect('/admin')
        else:
            return "Invalid Login"

    return render_template("admin_login.html")

@app.route('/logout')
@app.route('/admin_logout')
def logout():
    session.pop('admin',None)
    return redirect('/')

@app.route('/student_logout')
def student_logout():
    session.pop('student',None)
    return redirect('/')

@app.route('/admin')
def admin():
    if 'admin' not in session:
        return redirect('/admin_login')

    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()

    cursor.execute("SELECT * FROM teachers")
    teachers = cursor.fetchall()

    cursor.execute("""
        SELECT holiday_date, reason
        FROM admin_holidays
        WHERE holiday_date >= CURDATE()
        ORDER BY holiday_date ASC
    """)
    admin_holidays = cursor.fetchall()
    admin_holiday_dates = _get_admin_holiday_dates()

    timetable_error = session.pop('timetable_error', None)
    student_error = session.pop('student_error', None)
    holiday_message = session.pop('holiday_message', None)
    return render_template(
        "admin.html",
        students=students,
        teachers=teachers,
        admin_holidays=admin_holidays,
        admin_holiday_dates=admin_holiday_dates,
        timetable_error=timetable_error,
        student_error=student_error,
        holiday_message=holiday_message,
    )

# ---------------- ADMIN OD REQUESTS ----------------
@app.route('/admin_od_requests')
def admin_od_requests():
    if 'admin' not in session:
        return redirect('/admin_login')
    
    cursor.execute("""
        SELECT o.id, o.student_id, s.name, o.date, o.subject, o.hour, o.reason, o.status, o.created_at
        FROM od_requests o
        JOIN students s ON o.student_id = s.id
        WHERE o.status = 'Pending'
        ORDER BY o.created_at DESC
    """)
    pending_ods = cursor.fetchall()
    
    return render_template("admin_od.html", pending_ods=pending_ods)

@app.route('/approve_od/<int:request_id>', methods=['POST'])
def approve_od(request_id):
    if 'admin' not in session:
        return redirect('/admin_login')
        
    cursor.execute("SELECT student_id, date, subject, hour FROM od_requests WHERE id=%s", (request_id,))
    od_req = cursor.fetchone()
    
    if od_req:
        student_id, req_date, req_subject, req_hour = od_req
        
        # Update attendance
        cursor.execute("""
            UPDATE attendance 
            SET status='Present' 
            WHERE student_id=%s AND date=%s AND subject=%s AND hour=%s
        """, (student_id, req_date, req_subject, req_hour))
        
        # Update od_requests
        cursor.execute("UPDATE od_requests SET status='Approved' WHERE id=%s", (request_id,))
        db.commit()
        
    return redirect('/admin_od_requests')

@app.route('/reject_od/<int:request_id>', methods=['POST'])
def reject_od(request_id):
    if 'admin' not in session:
        return redirect('/admin_login')
        
    cursor.execute("UPDATE od_requests SET status='Rejected' WHERE id=%s", (request_id,))
    db.commit()
    
    return redirect('/admin_od_requests')


# ADD STUDENT
@app.route('/add_student',methods=['POST'])
def add_student():
    if 'admin' not in session:
        return redirect('/admin_login')

    id=request.form['id']
    name=request.form['name']
    batch=request.form['batch']
    sem=request.form['sem']
    email=request.form['email']
    parent=request.form['parent']
    password=request.form['password']
    parent_phone=request.form.get('parent_phone', '').strip()

    cursor.execute("SELECT id, name FROM students WHERE id=%s", (id,))
    if cursor.fetchone():
        session['student_error'] = f"A student with roll number {id} already exists."
        return redirect('/admin')

    cursor.execute(
        "SELECT id, name FROM students WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s))",
        (name,),
    )
    existing = cursor.fetchone()
    if existing:
        existing_id, existing_name = existing
        session['student_error'] = (
            f"A student named {existing_name} already exists (Roll No: {existing_id}). "
            f"You cannot register the same student again with a different roll number."
        )
        return redirect('/admin')

    photo=request.files['photo']
    photo_extra=request.files.get('photo_extra')

    temp_paths = []
    try:
        suffix = os.path.splitext(photo.filename or '.jpg')[1] or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            photo.save(tmp.name)
            temp_paths.append(tmp.name)
            photo_match = _find_existing_student_by_photo(tmp.name)
            if photo_match:
                existing_id, existing_name = photo_match
                session['student_error'] = (
                    f"This photo belongs to {existing_name} (Roll No: {existing_id}). "
                    f"The same person cannot be registered again with a different roll number."
                )
                return redirect('/admin')

        if photo_extra and photo_extra.filename:
            suffix_extra = os.path.splitext(photo_extra.filename)[1] or '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_extra) as tmp_extra:
                photo_extra.save(tmp_extra.name)
                temp_paths.append(tmp_extra.name)
                photo_match = _find_existing_student_by_photo(tmp_extra.name)
                if photo_match:
                    existing_id, existing_name = photo_match
                    session['student_error'] = (
                        f"This photo belongs to {existing_name} (Roll No: {existing_id}). "
                        f"The same person cannot be registered again with a different roll number."
                    )
                    return redirect('/admin')
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

    photo.seek(0)
    filename=id+".jpg"
    photo.save(os.path.join("dataset",filename))

    if photo_extra and photo_extra.filename:
        photo_extra.seek(0)
        filename_extra=id+"_extra.jpg"
        photo_extra.save(os.path.join("dataset",filename_extra))

    cursor.execute(
        "INSERT INTO students (id, name, batch, semester, email, parent_email, image, password, parent_phone) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (id,name,batch,sem,email,parent,filename,password,parent_phone)
    )
    db.commit()

    try:
        refresh_face_encoding_cache()
    except Exception:
        pass

    return redirect('/admin')

# DELETE STUDENT
@app.route('/delete_student/<id>')
def delete_student(id):
    cursor.execute("DELETE FROM students WHERE id=%s",(id,))
    db.commit()

    try:
        refresh_face_encoding_cache()
    except Exception:
        pass

    return redirect('/admin')

@app.route('/student_login',methods=['GET','POST'])
def student_login():
    if request.method=="POST":
        sid=request.form['id']
        password=request.form['password']

        cursor.execute(
            "SELECT * FROM students WHERE id=%s AND password=%s",
            (sid,password)
        )
        student=cursor.fetchone()

        if student:
            session['student']=sid
            return redirect('/student_dashboard')
        else:
            return "Invalid Student ID or Password"

    return render_template("student_login.html")

# ---------------- STUDENT DASHBOARD ----------------
@app.route('/student_dashboard')
def student_dashboard():
    if 'student' not in session:
        return redirect('/student_login')

    sid=session['student']

    cursor.execute("SELECT * FROM students WHERE id=%s",(sid,))
    student=cursor.fetchone()

    percentage=calculate_percentage(sid)

    cursor.execute("""
        SELECT subject,
        ROUND(COUNT(CASE WHEN status='Present' THEN 1 END)*100/COUNT(*),2)
        FROM attendance
        WHERE student_id=%s
        GROUP BY subject
    """,(sid,))
    attendance=cursor.fetchall()

    return render_template("student.html",
                           student=student,
                           attendance=attendance,
                           percentage=percentage)

@app.route('/apply_od', methods=['GET', 'POST'])
def apply_od():
    if 'student' not in session:
        return redirect('/student_login')
    
    sid = session['student']
    
    if request.method == 'POST':
        date = request.form['date']
        subject = request.form['subject']
        hour = request.form['hour']
        reason = request.form['reason']
        
        # Check if request already exists
        cursor.execute("SELECT id FROM od_requests WHERE student_id=%s AND date=%s AND subject=%s AND hour=%s", 
                      (sid, date, subject, hour))
        if cursor.fetchone():
            return "OD Request already submitted for this slot."
            
        cursor.execute("""
            INSERT INTO od_requests (student_id, date, subject, hour, reason)
            VALUES (%s, %s, %s, %s, %s)
        """, (sid, date, subject, hour, reason))
        db.commit()
        return redirect('/apply_od')
        
    # GET method: fetch absents and OD history
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')
    
    # Default to_date to current date if not provided
    from datetime import date
    if not to_date:
        to_date = date.today().strftime('%Y-%m-%d')
    
    query = """
        SELECT a.date, a.subject, a.hour 
        FROM attendance a
        LEFT JOIN od_requests o 
          ON a.student_id = o.student_id 
          AND a.date = o.date 
          AND a.subject = o.subject 
          AND a.hour = o.hour
        WHERE a.student_id=%s AND a.status='Absent' AND o.id IS NULL
    """
    params = [sid]
    
    if from_date:
        query += " AND a.date >= %s"
        params.append(from_date)
        
    if to_date:
        query += " AND a.date <= %s"
        params.append(to_date)
        
    query += " ORDER BY a.date DESC"
    
    cursor.execute(query, tuple(params))
    absent_records = cursor.fetchall()
    
    cursor.execute("""
        SELECT date, subject, hour, reason, status, created_at 
        FROM od_requests 
        WHERE student_id=%s 
        ORDER BY created_at DESC
    """, (sid,))
    od_history = cursor.fetchall()
    
    return render_template("apply_od.html", absent_records=absent_records, od_history=od_history, from_date=from_date, to_date=to_date)

# ---------------- PROFILE EDIT REQUESTS ----------------
@app.route('/request_profile_edit', methods=['POST'])
def request_profile_edit():
    if 'student' not in session:
        return redirect('/student_login')
    sid = session['student']
    cursor.execute("UPDATE students SET profile_edit_status='pending' WHERE id=%s", (sid,))
    db.commit()
    return redirect('/change_profile')

@app.route('/admin_approve_profile_edit/<sid>')
def admin_approve_profile_edit(sid):
    if 'admin' not in session:
        return redirect('/admin_login')
    cursor.execute("UPDATE students SET profile_edit_status='approved' WHERE id=%s", (sid,))
    db.commit()
    return redirect('/admin')

@app.route('/admin_reject_profile_edit/<sid>')
def admin_reject_profile_edit(sid):
    if 'admin' not in session:
        return redirect('/admin_login')
    cursor.execute("UPDATE students SET profile_edit_status='none' WHERE id=%s", (sid,))
    db.commit()
    return redirect('/admin')

#---------------- CHANGE PROFILE ----------------
@app.route('/change_profile', methods=['GET', 'POST'])
def change_profile():
    if 'student' not in session:
        return redirect('/student_login')

    sid = session['student']
    cursor.execute("SELECT * FROM students WHERE id=%s", (sid,))
    student = cursor.fetchone()

    success = None
    error = None

    if request.method == 'POST':
        # Existing profile edit handling
        # After successful photo verification and DB update, invalidate cache for this student
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip()
        parent   = request.form.get('parent', '').strip()
        parent_phone = request.form.get('parent_phone', '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm_password', '').strip()
        photo       = request.files.get('photo')
        photo_extra = request.files.get('photo_extra')

        photo_uploaded = (photo and photo.filename != '')
        extra_uploaded = (photo_extra and photo_extra.filename != '')

        # Enforce approved status ONLY for photo changes
        profile_edit_status = student[9] if len(student) > 9 else 'none'
        if (photo_uploaded or extra_uploaded) and profile_edit_status != 'approved':
            error = "Photo editing is locked. Please request approval from the administrator."
            return render_template("change_profile.html", student=student, success=success, error=error)

        # Validate password match
        if password and password != confirm:
            error = "Passwords do not match."
        else:
            temp_photo_path = None
            temp_extra_path = None
            verified_photo_uploaded = False
            verified_extra_uploaded = False

            # Check face match for primary photo if uploaded
            if photo_uploaded:
                temp_filename = f"{sid}_temp.jpg"
                temp_photo_path = os.path.join("dataset", temp_filename)
                photo.save(temp_photo_path)
                
                matches, match_err = _uploaded_photo_matches_student(sid, temp_photo_path)
                if not matches:
                    error = match_err
                    if os.path.exists(temp_photo_path):
                        os.remove(temp_photo_path)
                    return render_template("change_profile.html", student=student, success=success, error=error)
                verified_photo_uploaded = True

            # Check face match for extra photo if uploaded
            if extra_uploaded:
                temp_extra_filename = f"{sid}_extra_temp.jpg"
                temp_extra_path = os.path.join("dataset", temp_extra_filename)
                photo_extra.save(temp_extra_path)
                
                matches, match_err = _uploaded_photo_matches_student(sid, temp_extra_path)
                if not matches:
                    error = match_err
                    if temp_photo_path and os.path.exists(temp_photo_path):
                        os.remove(temp_photo_path)
                    if os.path.exists(temp_extra_path):
                        os.remove(temp_extra_path)
                    return render_template("change_profile.html", student=student, success=success, error=error)
                verified_extra_uploaded = True

            # Rename temp files to final files now that they are verified
            if verified_photo_uploaded and temp_photo_path:
                final_filename = f"{sid}.jpg"
                final_photo_path = os.path.join("dataset", final_filename)
                if os.path.exists(final_photo_path):
                    os.remove(final_photo_path)
                os.rename(temp_photo_path, final_photo_path)

            if verified_extra_uploaded and temp_extra_path:
                final_extra_filename = f"{sid}_extra.jpg"
                final_extra_path = os.path.join("dataset", final_extra_filename)
                if os.path.exists(final_extra_path):
                    os.remove(final_extra_path)
                os.rename(temp_extra_path, final_extra_path)

            # Determine next edit status
            next_status = 'none' if (photo_uploaded or extra_uploaded) else profile_edit_status

            # Build update query dynamically
            if password:
                cursor.execute(
                    "UPDATE students SET name=%s, email=%s, parent_email=%s, parent_phone=%s, password=%s, profile_edit_status=%s WHERE id=%s",
                    (name, email, parent, parent_phone, password, next_status, sid)
                )
            else:
                cursor.execute(
                    "UPDATE students SET name=%s, email=%s, parent_email=%s, parent_phone=%s, profile_edit_status=%s WHERE id=%s",
                    (name, email, parent, parent_phone, next_status, sid)
                )
            db.commit()

            # Refresh student data
            cursor.execute("SELECT * FROM students WHERE id=%s", (sid,))
            student = cursor.fetchone()
            success = "Profile updated successfully!"

            # Refresh face encoding cache if photos changed
            if photo_uploaded or extra_uploaded:
                try:
                    invalidate_face_cache(sid)
                except Exception as e:
                    print('Cache invalidation error:', e)
                try:
                    refresh_face_encoding_cache()
                except Exception:
                    pass

    return render_template("change_profile.html", student=student, success=success, error=error)

@app.route('/students_batch/<batch>')
def students_batch(batch):
    cursor.execute("SELECT * FROM students")
    data = cursor.fetchall()

    target_norm = _normalize_batch_value(batch)
    matched_data = []
    for row in data:
        # row[2] is batch in students table
        if _normalize_batch_value(row[2]) == target_norm:
            matched_data.append(row)

    return render_template("batch_students.html", data=matched_data, batch=batch)

@app.route('/student', methods=['GET', 'POST'])
def student():
    if request.method == "POST":
        sid = request.form['sid']
    else:
        # GET request fallback: redirect to student dashboard if logged in
        if 'student' in session:
            return redirect('/student_dashboard')
        return redirect('/student_login')

    cursor.execute("SELECT * FROM students WHERE id=%s", (sid,))
    student = cursor.fetchone()

    if student is None:
        return "Student not found"

    query = """
    SELECT subject,
    ROUND(COUNT(CASE WHEN status='Present' THEN 1 END)*100/COUNT(*),2)
    FROM attendance
    WHERE student_id=%s
    GROUP BY subject
    """

    cursor.execute(query, (sid,))
    attendance = cursor.fetchall()

    percentage = calculate_percentage(sid)

    if percentage < 75:
        subject_line = "Low Attendance Alert"
        body_text = f"Your child's Attendance percentage is {percentage}%.\nMinimum required attendance is 75%."
        ok, err = send_alert(student[5], subject_line, body_text)
        if not ok:
            print("Email sending failed:", err)
        
        if len(student) > 8 and student[8]:
            ok_sms, err_sms = send_sms(student[8], body_text)
            if not ok_sms:
                print("SMS sending failed:", err_sms)

    return render_template("student.html",
                           student=student,
                           attendance=attendance,
                           percentage=percentage)

@app.route('/edit_student/<id>', methods=['GET', 'POST'])
def edit_student(id):
    if request.method == "POST":
        name = request.form['name']
        batch = request.form['batch']
        sem = request.form['sem']
        email = request.form['email']
        parent_phone = request.form.get('parent_phone', '')

        cursor.execute(
            "UPDATE students SET name=%s,batch=%s,semester=%s,email=%s,parent_phone=%s WHERE id=%s",
            (name, batch, sem, email, parent_phone, id)
        )
        db.commit()
        return redirect('/admin')

    cursor.execute("SELECT * FROM students WHERE id=%s", (id,))
    student = cursor.fetchone()

    return render_template("edit_student.html", student=student)

@app.route('/admin_view_student/<id>')
def admin_view_student(id):
    if 'admin' not in session:
        return redirect('/admin_login')

    cursor.execute("SELECT * FROM students WHERE id=%s", (id,))
    student = cursor.fetchone()

    if not student:
        return "Student not found"

    query = """
    SELECT subject,
    ROUND(COUNT(CASE WHEN status='Present' THEN 1 END)*100/COUNT(*),2)
    FROM attendance
    WHERE student_id=%s
    GROUP BY subject
    """

    cursor.execute(query, (id,))
    attendance = cursor.fetchall()

    percentage = calculate_percentage(id)

    return render_template("admin_student_view.html",
                           student=student,
                           attendance=attendance,
                           percentage=percentage)

#---------------- CALCULATE ATTENDANCE ----------------
def calculate_percentage(student_id):
    cursor.execute("""
    SELECT COUNT(*), COUNT(CASE WHEN status='Present' THEN 1 END)
    FROM attendance
    WHERE student_id=%s
    """, (student_id,))
    total, present = cursor.fetchone()

    if not total:
        return 0
    return round((present / total) * 100, 2)

#---------------- EMAIL ALERT ----------------
#Mail only works if MAIL_SMTP_HOST + MAIL_USERNAME + MAIL_PASSWORD match your provider.
#Gmail / Google Workspace: smtp.gmail.com:587 + TLS + App Password (not your normal password).
#Microsoft 365: smtp.office365.com:587. Other hosts: ask your IT for SMTP settings.
def _format_smtp_exception(exc):
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        detail = exc.smtp_error
        if isinstance(detail, (bytes, bytearray)):
            detail = detail.decode("utf-8", errors="replace")
        return f"SMTP authentication failed: {exc.smtp_code} {detail or exc}"
    if isinstance(exc, smtplib.SMTPException):
        return f"SMTP error: {exc}"
    return f"{type(exc).__name__}: {exc}"

def send_alert(parent_email, subject_line, body_text):
    """
    Send one email. Returns (True, None) on success, (False, error_message) on failure.
    Configure via environment variables (see comments above this function).
    """
    to_addr = (parent_email or "").strip()
    if not to_addr or "@" not in to_addr:
        return False, "Invalid or empty recipient address"

    sender ="imca-443@scmsgroup.org"
    password = "axer udei spdu yxji"
    host = os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("MAIL_SMTP_PORT", "587"))
    use_tls = os.environ.get("MAIL_USE_TLS", "true").lower() in ("1", "true", "yes", "on")
    use_ssl = os.environ.get("MAIL_SMTP_SSL", "").lower() in ("1", "true", "yes", "on")

    if not password:
        return False, "MAIL_PASSWORD is not set (set it in Windows environment or .env)"

    msg = EmailMessage()
    msg["Subject"] = subject_line
    msg["From"] = sender
    msg["To"] = to_addr
    msg.set_content(body_text, charset="utf-8")

    try:
        timeout = int(os.environ.get("MAIL_SMTP_TIMEOUT", "30"))
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, timeout=timeout, context=context) as server:
                server.login(sender, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as server:
                server.ehlo()
                if use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                    server.ehlo()
                server.login(sender, password)
                server.send_message(msg)
    except Exception as exc:
        return False, _format_smtp_exception(exc)
    return True, None

def send_sms(to_number, body_text):
    """
    Send an SMS using Twilio.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER in environment.
    """
    if not to_number:
        return False, "No phone number provided"

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE_NUMBER")

    # If credentials are not configured, empty, or placeholders, use mock sending
    if (not account_sid or account_sid.strip() == "" or "your_account" in account_sid or
        not auth_token or auth_token.strip() == "" or "your_auth" in auth_token):
        print(f"[MOCK SMS] To: {to_number} | Message: {body_text}")
        return True, "Mock SMS sent successfully (Twilio credentials missing)"

    # Clean to_number: Keep only digits and '+'
    clean_to = "".join(c for c in str(to_number).strip() if c.isdigit() or c == '+')
    # If it is 10 digits (common for Indian numbers), prepend +91
    if len(clean_to) == 10 and clean_to.isdigit():
        clean_to = "+91" + clean_to
    elif len(clean_to) == 12 and clean_to.startswith("91") and not clean_to.startswith("+"):
        clean_to = "+" + clean_to
    elif not clean_to.startswith("+"):
        clean_to = "+91" + clean_to

    # Clean from_number
    clean_from = (from_number or "").strip()
    if clean_from and not clean_from.startswith("+") and clean_from.isdigit():
        if len(clean_from) == 10:
            clean_from = "+91" + clean_from
        elif len(clean_from) == 12 and clean_from.startswith("91"):
            clean_from = "+" + clean_from

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body_text,
            from_=clean_from,
            to=clean_to
        )
        print(f"SMS sent successfully to {clean_to}. SID: {message.sid}")
        return True, None
    except Exception as e:
        print(f"SMS sending failed to {clean_to}: {e}")
        return False, str(e)

def _students_in_batch_as_dicts(batch_norm):
    """
    All columns as dicts. Normalizes both the requested batch and the student's batch
    so that formats like 'imca B9', 'B9', 'Batch 9' are matched as the same batch.
    """
    cursor.execute("SELECT * FROM students")
    cols = [c[0] for c in cursor.description]
    all_rows = [dict(zip(cols, r)) for r in cursor.fetchall()]

    target_norm = _normalize_batch_value(batch_norm)
    matched_rows = []
    for row in all_rows:
        if _normalize_batch_value(row.get('batch')) == target_norm:
            matched_rows.append(row)
            
    return matched_rows
def _guardian_email_for_alert(row):
    """
    Pick an address for parent/guardian alerts. Requires '@' (SMTP cannot use phone-only).
    Checks common column names; order prefers explicit parent email fields.
    """
    lower = {str(k).lower(): v for k, v in row.items() if v is not None}
    for pk in (
        "parent_email",
        "parentemail",
        "guardian_email",
        "parent_guardian_email",
        "father_email",
        "mother_email",
    ):
        v = lower.get(pk)
        if v is not None:
            s = str(v).strip()
            if "@" in s:
                return s
    v = lower.get("parent")
    if v is not None:
        s = str(v).strip()
        if "@" in s:
            return s
    for lk, v in lower.items():
        if "mail" in lk and lk != "email":
            s = str(v).strip()
            if "@" in s:
                return s
    v = lower.get("email")
    if v is not None:
        s = str(v).strip()
        if "@" in s:
            return s
    return ""

#---------------- TIMETABLE ----------------
@app.route('/timetable')
def timetable():
    if 'admin' not in session and 'teacher' not in session:
        return redirect('/teacher_login')

    is_teacher = 'teacher' in session
    teacher_name = None

    if is_teacher:
        tid = session['teacher']
        cursor.execute("SELECT name FROM teachers WHERE id=%s", (tid,))
        row = cursor.fetchone()
        if row:
            teacher_name = row[0]

    import datetime
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')

    if not to_date:
        to_date = str(datetime.date.today())

    day_dates = {}
    calc_from = from_date
    if not calc_from:
        t_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        calc_from = str(t_d - datetime.timedelta(days=t_d.weekday()))

    prev_week_from = ""
    prev_week_to = ""
    next_week_from = ""
    next_week_to = ""

    try:
        start_d = datetime.datetime.strptime(calc_from, "%Y-%m-%d").date()
        end_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        if not from_date:
            end_d = start_d + datetime.timedelta(days=6)

        curr_d = start_d
        while curr_d <= end_d and (curr_d - start_d).days < 7:
            day_name = curr_d.strftime("%A")
            if day_name not in day_dates:
                day_dates[day_name] = curr_d.strftime("%Y-%m-%d")
            curr_d += datetime.timedelta(days=1)

        prev_week_from = str(start_d - datetime.timedelta(days=7))
        prev_week_to = str(start_d - datetime.timedelta(days=1))
        next_week_from = str(start_d + datetime.timedelta(days=7))
        next_week_to = str(start_d + datetime.timedelta(days=13))
    except Exception as e:
        print("Date parse error:", e)

    if is_teacher and teacher_name:
        cursor.execute(
            "SELECT id, day, hour, subject, teacher, date, month FROM timetable WHERE LOWER(TRIM(teacher)) = LOWER(TRIM(%s))",
            (teacher_name,)
        )
    else:
        cursor.execute("SELECT id, day, hour, subject, teacher, date, month FROM timetable")
    rows = cursor.fetchall()

    timetable = {}

    timetable_hours = {
        1: "08:30 - 09:30",
        2: "09:30 - 10:30",
        3: "10:45 - 11:45",
        4: "11:45 - 12:45",
        5: "13:45 - 14:45",
        6: "14:45 - 15:45",
        7: "15:45 - 16:45"
    }

    for t_id, day, hour, subject, teacher, t_date, month in rows:
        hour = int(hour)
        
        if day in day_dates:
            if str(t_date) != day_dates[day]:
                continue
        else:
            continue

        if day not in timetable:
            timetable[day] = {}

        timetable[day][hour] = {
            "id": t_id,
            "subject": subject,
            "teacher": teacher,
            "time": timetable_hours.get(hour, ""),
            "month": month
        }

    print("TIMETABLE DICT:", timetable)

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hours = [1,2,3,4,5]

    # Determine the display month(s)
    display_months = []
    for d_str in day_dates.values():
        try:
            m_y = datetime.datetime.strptime(d_str, "%Y-%m-%d").date().strftime("%B %Y")
            if m_y not in display_months:
                display_months.append(m_y)
        except Exception:
            pass
    current_month = " - ".join(display_months) if display_months else datetime.date.today().strftime("%B %Y")

    admin_holiday_dates = _get_admin_holiday_dates()

    return render_template(
        "timetable.html",
        timetable=timetable,
        days=days,
        hours=hours,
        day_dates=day_dates,
        from_date=from_date,
        to_date=to_date,
        is_teacher=is_teacher,
        teacher_name=teacher_name,
        current_month=current_month,
        prev_week_from=prev_week_from,
        prev_week_to=prev_week_to,
        next_week_from=next_week_from,
        next_week_to=next_week_to,
        admin_holiday_dates=admin_holiday_dates,
    )

@app.route('/mark_holiday', methods=['POST'])
def mark_holiday():
    if 'admin' not in session:
        return redirect('/admin_login')

    date_str = request.form.get('date', '').strip()
    reason = request.form.get('reason', '').strip() or 'Holiday'

    if not date_str:
        session['timetable_error'] = "Please select a date to mark as holiday."
        return redirect('/admin#holiday-section')

    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        session['timetable_error'] = "Invalid date selected."
        return redirect('/admin#holiday-section')

    cursor.execute(
        """
        INSERT INTO admin_holidays (holiday_date, reason)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE reason=%s
        """,
        (date_str, reason, reason),
    )
    cursor.execute("DELETE FROM timetable WHERE date=%s", (date_str,))
    db.commit()

    session['holiday_message'] = f"{date_str} marked as holiday ({reason}). All classes for that date were removed."
    return redirect('/admin#holiday-section')

@app.route('/remove_holiday/<date_str>')
def remove_holiday(date_str):
    if 'admin' not in session:
        return redirect('/admin_login')

    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        session['timetable_error'] = "Invalid holiday date."
        return redirect('/admin#holiday-section')

    cursor.execute("DELETE FROM admin_holidays WHERE holiday_date=%s", (date_str,))
    db.commit()
    session['holiday_message'] = f"Holiday removed for {date_str}. You can add classes for this date again."
    return redirect('/admin#holiday-section')

@app.route('/add_timetable', methods=['POST'])
def add_timetable():
    if 'admin' not in session:
        return redirect('/admin_login')

    day=request.form['day']
    hour=int(request.form['hour'])
    subject=request.form['subject']
    teacher=request.form['teacher']
    semester=request.form['semester']
    batch=request.form.get('batch', '')
    date=request.form.get('date', '')
    month=request.form.get('month', '')

    if date:
        try:
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            if is_holiday(date_obj):
                admin_holidays = _get_admin_holiday_dates()
                if date in admin_holidays:
                    session['timetable_error'] = (
                        f"Cannot add timetable for {date} — it is marked as a holiday ({admin_holidays[date]})."
                    )
                else:
                    session['timetable_error'] = (
                        f"Cannot add timetable for {date} — 2nd/4th Saturdays and Sundays are holidays."
                    )
                return redirect('/admin#timetable-section')
        except Exception:
            pass

    if not month and date:
        try:
            month = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B")
        except Exception:
            pass

    cursor.execute(
        "INSERT INTO timetable(day,hour,subject,teacher,semester,batch,date,month) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
        (day,hour,subject,teacher,semester,batch,date,month)
    )
    db.commit()

    return redirect('/admin')

@app.route('/edit_timetable/<id>', methods=['GET', 'POST'])
def edit_timetable(id):
    if 'admin' not in session:
        return redirect('/admin_login')

    if request.method == "POST":
        day = request.form['day']
        hour = int(request.form['hour'])
        subject = request.form['subject']
        teacher = request.form['teacher']
        semester = request.form['semester']
        batch = request.form.get('batch', '')
        date = request.form.get('date', '')
        month = request.form.get('month', '')

        if date:
            try:
                date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
                if is_holiday(date_obj):
                    admin_holidays = _get_admin_holiday_dates()
                    if date in admin_holidays:
                        session['timetable_error'] = (
                            f"Cannot set date {date} — it is marked as a holiday ({admin_holidays[date]})."
                        )
                    else:
                        session['timetable_error'] = (
                            f"Cannot set date {date} — 2nd/4th Saturdays and Sundays are holidays."
                        )
                    return redirect(f'/edit_timetable/{id}')
            except Exception:
                pass

        if not month and date:
            try:
                month = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%B")
            except Exception:
                pass

        cursor.execute(
            "UPDATE timetable SET day=%s, hour=%s, subject=%s, teacher=%s, semester=%s, batch=%s, date=%s, month=%s WHERE id=%s",
            (day, hour, subject, teacher, semester, batch, date, month, id)
        )
        db.commit()
        return redirect('/timetable')

    cursor.execute("SELECT * FROM timetable WHERE id=%s", (id,))
    entry = cursor.fetchone()
    cursor.execute("SELECT * FROM teachers")
    teachers = cursor.fetchall()
    timetable_error = session.pop('timetable_error', None)
    admin_holiday_dates = _get_admin_holiday_dates()
    return render_template(
        "edit_timetable.html",
        entry=entry,
        teachers=teachers,
        timetable_error=timetable_error,
        admin_holiday_dates=admin_holiday_dates,
    )

@app.route('/delete_timetable/<id>')
def delete_timetable(id):
    if 'admin' not in session:
        return redirect('/admin_login')

    cursor.execute("DELETE FROM timetable WHERE id=%s", (id,))
    db.commit()
    return redirect('/timetable')

@app.route('/view_timetable')
def view_timetable():
    cursor.execute("SELECT day,hour,subject,teacher FROM timetable")
    rows = cursor.fetchall()

    timetable = {}

    for day, hour, subject, teacher in rows:
        hour = int(hour)

        if day not in timetable:
            timetable[day] = {}

        timetable[day][hour] = subject + "<br>" + teacher

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    hours = [1,2,3,4,5]

    return render_template("view_timetable.html", timetable=timetable, days=days, hours=hours)
@app.route('/student_timetable')
def student_timetable():
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')

    if not to_date:
        to_date = str(datetime.date.today())

    if 'student' not in session:
        return redirect('/student_login')

    sid = session['student']

    timetable_hours = {
        1: "08:30 - 09:30",
        2: "09:30 - 10:30",
        3: "10:45 - 11:45",
        4: "11:45 - 12:45",
        5: "13:45 - 14:45",
        6: "14:45 - 15:45",
        7: "15:45 - 16:45"
    }




    admin_holiday_dates = _get_admin_holiday_dates()

    


    

    # Calculate dates for each day of the week based on the selected range
    day_dates = {}
    calc_from = from_date
    if not calc_from:
        t_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        calc_from = str(t_d - datetime.timedelta(days=t_d.weekday()))
        
    prev_week_from = ""
    prev_week_to = ""
    next_week_from = ""
    next_week_to = ""

    try:
        start_d = datetime.datetime.strptime(calc_from, "%Y-%m-%d").date()
        end_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        if not from_date:
            end_d = start_d + datetime.timedelta(days=6)
            
        curr_d = start_d
        while curr_d <= end_d and (curr_d - start_d).days < 7:
            day_name = curr_d.strftime("%A")
            if day_name not in day_dates:
                day_dates[day_name] = curr_d.strftime("%Y-%m-%d")
            curr_d += datetime.timedelta(days=1)

        prev_week_from = str(start_d - datetime.timedelta(days=7))
        prev_week_to = str(start_d - datetime.timedelta(days=1))
        next_week_from = str(start_d + datetime.timedelta(days=7))
        next_week_to = str(start_d + datetime.timedelta(days=13))
    except Exception as e:
        print("Date parse error:", e)

    # Get timetable
    cursor.execute("SELECT day,hour,subject,teacher,date,month FROM timetable")
    rows = cursor.fetchall()

    timetable = {}
    for day,hour_str,subject,teacher,t_date,month in rows:
        hour = int(hour_str)

        if day in day_dates:
            if str(t_date) != day_dates[day]:
                continue
        else:
            continue

        if day not in timetable:
            timetable[day] = {}
        
        timetable[day][hour] = {
            "subject": subject,
            "teacher": teacher,
            "time": timetable_hours.get(hour, ""),
            "month": month
        }

    # 📊 Attendance based on date filter (uses calc_from so that the default weekly view displays attendance!)
    attendance = {}
    if calc_from and to_date:
        cursor.execute("""
        SELECT subject, status, date, hour 
        FROM attendance 
        WHERE student_id=%s AND date BETWEEN %s AND %s
        """, (sid, calc_from, to_date))

        att_rows = cursor.fetchall()

        for subject, status, date, hour_db in att_rows:
            hour_slot = None
            if hour_db is not None:
                try:
                    hour_slot = int(hour_db)
                except (TypeError, ValueError):
                    pass
            
            # Fallback to matching by subject name if hour was not recorded
            if not hour_slot:
                cal_day = date.strftime("%A")
                if cal_day in timetable:
                    for h, slot_info in timetable[cal_day].items():
                        if slot_info['subject'] == subject:
                            hour_slot = h
                            break
                        
            if hour_slot:
                # 1. Try to map to the calendar day first
                cal_day = date.strftime("%A")
                if cal_day in timetable and hour_slot in timetable[cal_day] and timetable[cal_day][hour_slot]['subject'] == subject:
                    if cal_day not in attendance:
                        attendance[cal_day] = {}
                    attendance[cal_day][hour_slot] = status
                else:
                    # 2. Search other days if calendar day doesn't have this scheduled slot (useful for make-up classes/practice tests)
                    matched_day = None
                    for day_name in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
                        if day_name in timetable and hour_slot in timetable[day_name]:
                            if timetable[day_name][hour_slot]['subject'] == subject:
                                matched_day = day_name
                                break
                    if matched_day:
                        if matched_day not in attendance:
                            attendance[matched_day] = {}
                        attendance[matched_day][hour_slot] = status

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    # Determine the display month(s)
    display_months = []
    for d_str in day_dates.values():
        try:
            m_y = datetime.datetime.strptime(d_str, "%Y-%m-%d").date().strftime("%B %Y")
            if m_y not in display_months:
                display_months.append(m_y)
        except Exception:
            pass
    current_month = " - ".join(display_months) if display_months else datetime.date.today().strftime("%B %Y")

    return render_template(
        "student_timetable.html",
        timetable=timetable,
        attendance=attendance,
        days=days,
        from_date=from_date,
        to_date=to_date,
        day_dates=day_dates,
        current_month=current_month,
        prev_week_from=prev_week_from,
        prev_week_to=prev_week_to,
        next_week_from=next_week_from,
        next_week_to=next_week_to,
        admin_holiday_dates=admin_holiday_dates
    )

# ---------------- TEACHER PANEL ----------------
@app.route('/teacher_login',methods=['GET','POST'])
def teacher_login():
    if request.method=="POST":
        tid=request.form['id']
        password=request.form['password']

        cursor.execute("SELECT * FROM teachers WHERE id=%s AND password=%s",(tid,password))
        teacher=cursor.fetchone()

        if teacher:
            session['teacher']=tid
            return redirect('/teacher')

    return render_template("teacher_login.html")










#-------------------AI LEAVE ADVISOR--------------------------#
@app.route('/attendance_chatbot', methods=['POST'])
def attendance_chatbot():
    data = request.get_json()
    question = data['message'].lower()

    student_id = session['student']

    # Month-wise attendance first
    if "month" in question:
        rows = monthwise_attendance(student_id)
        reply = "📅 Month-wise Attendance\n\n"
        for month, pct in rows:
            reply += f"{month}: {pct}%\n"
        return {"reply": reply}

    # Overall attendance
    elif "attendance" in question:
        pct = get_attendance_percentage(student_id)
        return {
            "reply": f"📊 Your current attendance is {pct}%"
        }

    elif "leave" in question or "miss" in question:
        pct = get_attendance_percentage(student_id)
        safe = leave_recommendation(student_id)

        if pct < 75:
            return {
                "reply":
                f"🔴 ATTENDANCE STATUS: CRITICAL\n\n"
                f"📊 Current Attendance: {pct}%\n\n"
                f"❌ You are already below the required 75% attendance.\n"
                f"🚫 Taking leave is not recommended.\n"
                f"📚 Attend upcoming classes regularly to improve your attendance."
            }
        elif safe > 0:
            return {
                "reply":
                f"🟢 ATTENDANCE STATUS: SAFE\n\n"
                f"📊 Current Attendance: {pct}%\n"
                f"🏖 You can safely miss {safe} more classes while maintaining 75% attendance."
            }
        else:
            return {
                "reply":
                f"🟡 ATTENDANCE STATUS: WARNING\n\n"
                f"📊 Current Attendance: {pct}%\n"
                f"⚠ You are exactly at the attendance limit.\n"
                f"❌ No more leave is recommended."
            }
    
def get_attendance_percentage(student_id):
    cursor.execute("""
        SELECT COUNT(*), COUNT(CASE WHEN status='Present' THEN 1 END)
        FROM attendance
        WHERE student_id=%s
    """, (student_id,))
    total, present = cursor.fetchone()

    if not total:
        return 0

    return round((present / total) * 100, 2)

def monthwise_attendance(student_id):
    cursor.execute("""
        SELECT
            MONTHNAME(date),
            ROUND(
                SUM(
                    CASE
                    WHEN status='Present'
                    THEN 1
                    ELSE 0
                    END
                ) *100.0 / COUNT(*),
                2
            )
        FROM attendance
        WHERE student_id=%s
        GROUP BY MONTH(date)
        ORDER BY MONTH(date)
    """,(student_id,))

    return cursor.fetchall()

def leave_recommendation(student_id):
    cursor.execute("""
        SELECT COUNT(*), COUNT(CASE WHEN status='Present' THEN 1 END)
        FROM attendance
        WHERE student_id=%s
    """, (student_id,))
    total, present = cursor.fetchone()

    if not total:
        return 0

    leaves = 0
    while ((present/(total+leaves))*100) >= 75:
        leaves += 1

    return max(0, leaves-1)

def predict_after_leave(student_id, missed_classes):
    cursor.execute("""
        SELECT COUNT(*), COUNT(CASE WHEN status='Present' THEN 1 END)
        FROM attendance
        WHERE student_id=%s
    """, (student_id,))
    total, present = cursor.fetchone()
    total = total or 0
    present = present or 0

    future = round(
        (present/(total+missed_classes))*100,
        2
    )
    return future

@app.route('/teacher_logout')
def teacher_logout():
    session.pop('teacher', None)
    return redirect('/')

@app.route('/teacher')
def teacher():
    if 'teacher' not in session:
        return redirect('/teacher_login')

    tid = session['teacher']
    cursor.execute("SELECT * FROM teachers WHERE id=%s", (tid,))
    teacher_row = cursor.fetchone()
    if not teacher_row:
        session.pop('teacher', None)
        return redirect('/teacher_login')

    teacher_name = teacher_row[1]

    import datetime
    from_date = request.args.get('from_date', '')
    to_date = request.args.get('to_date', '')

    if not to_date:
        to_date = str(datetime.date.today())

    day_dates = {}
    calc_from = from_date
    if not calc_from:
        # Default to Monday of the week containing to_date
        t_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        calc_from = str(t_d - datetime.timedelta(days=t_d.weekday()))
        
    prev_week_from = ""
    prev_week_to = ""
    next_week_from = ""
    next_week_to = ""

    try:
        start_d = datetime.datetime.strptime(calc_from, "%Y-%m-%d").date()
        end_d = datetime.datetime.strptime(to_date, "%Y-%m-%d").date()
        if not from_date:
            end_d = start_d + datetime.timedelta(days=6) # 7 days
            
        curr_d = start_d
        while curr_d <= end_d and (curr_d - start_d).days < 7:
            day_name = curr_d.strftime("%A")
            if day_name not in day_dates:
                day_dates[day_name] = curr_d.strftime("%Y-%m-%d")
            curr_d += datetime.timedelta(days=1)

        prev_week_from = str(start_d - datetime.timedelta(days=7))
        prev_week_to = str(start_d - datetime.timedelta(days=1))
        next_week_from = str(start_d + datetime.timedelta(days=7))
        next_week_to = str(start_d + datetime.timedelta(days=13))
    except Exception as e:
        print("Date parse error:", e)

    cursor.execute(
        "SELECT id, day, hour, subject, teacher, semester, batch, date, month FROM timetable WHERE LOWER(TRIM(teacher)) = LOWER(TRIM(%s))",
        (teacher_name,),
    )
    all_classes = cursor.fetchall()

    date_today = datetime.date.today()
    date_yesterday = date_today - datetime.timedelta(days=1)

    cursor.execute("SELECT DISTINCT subject, date, hour FROM attendance")
    taken_attendance = [f"{str(row[0]).strip().lower()}-{str(row[1]).strip()}-{str(row[2]).strip()}" for row in cursor.fetchall()]

    classes = []
    seen_classes = set()
    for row in all_classes:
        day = row[1]
        t_date = row[7]

        if day in day_dates:
            if str(t_date) != day_dates[day]:
                continue
        else:
            continue

        # Get the date that will actually be displayed in the UI row
        class_date = day_dates.get(day, 'N/A')
        
        # Create a unique key for deduplication based on shown columns:
        # Date, Day, Hour, Subject, Semester, Batch
        unique_key = (
            str(class_date).strip(),
            str(day).strip(),
            str(row[2]).strip(),        # hour
            str(row[3]).strip().lower(), # subject
            str(row[5]).strip(),        # semester
            str(row[6]).strip().lower()  # batch
        )

        if unique_key in seen_classes:
            continue
        seen_classes.add(unique_key)
        classes.append(row)

    def _class_sort_key(row):
        class_date = day_dates.get(row[1], row[7])
        try:
            date_key = datetime.datetime.strptime(str(class_date), "%Y-%m-%d").date()
        except Exception:
            date_key = datetime.date.max
        try:
            hour_key = int(row[2])
        except Exception:
            hour_key = 0
        return (date_key, hour_key)

    classes.sort(key=_class_sort_key)

    # Determine the display month(s)
    display_months = []
    for d_str in day_dates.values():
        try:
            m_y = datetime.datetime.strptime(d_str, "%Y-%m-%d").date().strftime("%B %Y")
            if m_y not in display_months:
                display_months.append(m_y)
        except Exception:
            pass
    current_month = " - ".join(display_months) if display_months else datetime.date.today().strftime("%B %Y")

    return render_template(
        "teacher.html",
        classes=classes,
        day_dates=day_dates,
        from_date=from_date,
        to_date=to_date,
        teacher_name=teacher_name,
        taken_attendance=taken_attendance,
        date_today=str(date_today),
        date_yesterday=str(date_yesterday),
        current_month=current_month,
        prev_week_from=prev_week_from,
        prev_week_to=prev_week_to,
        next_week_from=next_week_from,
        next_week_to=next_week_to
    )

@app.route('/add_teacher',methods=['POST'])
def add_teacher():
    if 'admin' not in session:
        return redirect('/admin_login')

    id=request.form['id']
    name=request.form['name']
    subject=request.form['subject']
    email=request.form['email']
    password=request.form['password']
    batch=request.form.get('batch', '')

    cursor.execute(
        "INSERT INTO teachers VALUES(%s,%s,%s,%s,%s,%s)",
        (id,name,subject,email,password,batch)
    )
    db.commit()

    return redirect('/admin')
@app.route('/delete_teacher/<id>')
def delete_teacher(id):
    cursor.execute("DELETE FROM teachers WHERE id=%s",(id,))
    db.commit()
    return redirect('/admin')

@app.route('/edit_teacher/<id>',methods=['GET','POST'])
def edit_teacher(id):
    if request.method=="POST":
        name=request.form['name']
        subject=request.form['subject']
        email=request.form['email']
        batch=request.form['batch']
        password=request.form.get('password', '')

        if password.strip():
            cursor.execute(
                "UPDATE teachers SET name=%s,subject=%s,email=%s,batch=%s,password=%s WHERE id=%s",
                (name,subject,email,batch,password,id)
            )
        else:
            cursor.execute(
                "UPDATE teachers SET name=%s,subject=%s,email=%s,batch=%s WHERE id=%s",
                (name,subject,email,batch,id)
            )

        db.commit()
        return redirect('/admin')

    cursor.execute("SELECT * FROM teachers WHERE id=%s",(id,))
    teacher=cursor.fetchone()
    return render_template("edit_teacher.html",teacher=teacher)

# ---------------- FACE RECOGNITION ATTENDANCE ----------------
def get_current_subject():
    now = datetime.datetime.now()
    day = now.strftime("%A")
    current_time = now.time()

    timetable_hours = {
        1: ("08:30:00","09:30:00"),
        2: ("09:30:00","10:30:00"),
        3: ("10:45:00","11:45:00"),
        4: ("11:45:00","12:45:00"),
        5: ("13:45:00","14:45:00"),
        6: ("14:45:00","15:45:00"),
        7: ("15:45:00","16:45:00")
    }

    for hour,(start,end) in timetable_hours.items():
        start=datetime.datetime.strptime(start,"%H:%M:%S").time()
        end=datetime.datetime.strptime(end,"%H:%M:%S").time()

        if start<=current_time<=end:
            cursor.execute(
                "SELECT subject FROM timetable WHERE day=%s AND hour=%s",
                (day,hour)
            )
            data=cursor.fetchone()
            if data:
                return data[0]

    return "Unknown"

def _teacher_slot_allows_batch(teacher_name, day, hour_int, subject, batch_norm):
    """True if this teacher has a timetable row for the slot with matching batch."""
    cursor.execute(
        """
        SELECT COALESCE(batch, '') FROM timetable
        WHERE LOWER(TRIM(day)) = LOWER(TRIM(%s))
        AND hour = %s
        AND LOWER(TRIM(subject)) = LOWER(TRIM(%s))
        AND LOWER(TRIM(teacher)) = LOWER(TRIM(%s))
        """,
        (day, hour_int, subject, teacher_name),
    )
    target_norm = _normalize_batch_value(batch_norm)
    slot_batches = [_normalize_batch_value(r[0]) for r in cursor.fetchall()]
    if not slot_batches:
        return False
    return target_norm in slot_batches

def _attendance_slot_taken_in_db(subject, target_date, hour_int):
    cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE TRIM(subject)=TRIM(%s) AND date=%s AND hour=%s",
        (subject, target_date, str(hour_int)),
    )
    return cursor.fetchone()[0] > 0

def _draft_matches_slot(draft, subject, target_date, hour_int, batch):
    if not draft:
        return False
    try:
        hour_ok = int(draft.get("hour_int")) == int(hour_int)
    except (TypeError, ValueError):
        hour_ok = False
    return (
        draft.get("subject") == subject
        and draft.get("target_date") == target_date.isoformat()
        and hour_ok
        and _normalize_batch_value(draft.get("batch")) == _normalize_batch_value(batch)
    )

def _save_attendance_draft(draft):
    session["attendance_draft"] = draft
    session.modified = True

def _clear_attendance_draft():
    session.pop("attendance_draft", None)
    session.modified = True

def _student_lists_from_marked(student_dicts, marked_norm_ids):
    marked_norm = set()
    for m in marked_norm_ids:
        n = _norm_student_id(m)
        if n:
            marked_norm.add(n)
    present, absent = [], []
    for row in student_dicts:
        sid_norm = _norm_student_id(row.get("id"))
        info = {
            "id": row.get("id"),
            "name": (row.get("name") or "").strip() or str(row.get("id")),
        }
        if sid_norm in marked_norm:
            present.append(info)
        else:
            absent.append(info)
    return present, absent

def _finalize_attendance(student_dicts, marked_norm_ids, subject, target_date, hour_int, batch):
    """Write present/absent rows to DB and send absence emails (called only on Confirm)."""
    marked_norm = set()
    for m in marked_norm_ids:
        n = _norm_student_id(m)
        if n:
            marked_norm.add(n)
    ab = {
        "emails_sent": 0,
        "skipped_no_address": 0,
        "smtp_failed": 0,
        "skipped_present_today": 0,
        "existing_absent_resend": 0,
        "sms_sent": 0,
        "sms_failed": 0,
        "skipped_no_phone": 0,
    }
    smtp_error_samples = []
    sms_error_samples = []
    date = target_date
    hour_str = str(hour_int)

    for row in student_dicts:
        sid_key = _norm_student_id(row.get("id"))
        raw_id = row.get("id")
        student_name = (row.get("name") or "").strip() or str(sid_key or raw_id)
        parent_email = _guardian_email_for_alert(row)

        if sid_key is None:
            continue

        if sid_key in marked_norm:
            cursor.execute(
                """
                SELECT id FROM attendance
                WHERE CAST(student_id AS CHAR) = CAST(%s AS CHAR)
                  AND TRIM(subject) = TRIM(%s) AND date = %s AND hour = %s
                LIMIT 1
                """,
                (sid_key, subject, date, hour_str),
            )
            if not cursor.fetchone():
                now = datetime.datetime.now()
                cursor.execute(
                    "INSERT INTO attendance(student_id,subject,date,time,status,hour) VALUES(%s,%s,%s,%s,%s,%s)",
                    (raw_id, subject, date, now.time(), "Present", hour_str),
                )
                db.commit()
            continue

        cursor.execute(
            """
            SELECT id, status FROM attendance
            WHERE CAST(student_id AS CHAR) = CAST(%s AS CHAR)
              AND TRIM(subject) = TRIM(%s) AND date = %s AND hour = %s
            LIMIT 1
            """,
            (sid_key, subject, date, hour_str),
        )
        existing = cursor.fetchone()
        if existing:
            prev_status = (existing[1] or "").strip().lower()
            if prev_status == "present":
                ab["skipped_present_today"] += 1
                continue
            if prev_status == "absent":
                ab["existing_absent_resend"] += 1
        else:
            cursor.execute(
                "INSERT INTO attendance(student_id,subject,date,time,status,hour) VALUES(%s,%s,%s,%s,%s,%s)",
                (raw_id, subject, date, None, "Absent", hour_str),
            )
            db.commit()

        subject_line = f"Absence alert: {student_name} — {subject}"
        body_text = (
            f"This is to inform you that {student_name} (Student ID: {sid_key}, Batch: {batch}) "
            f"was marked ABSENT for {subject} on {date.strftime('%d-%m-%Y')} because their face "
            f"was not detected during batch attendance.\n\n"
            f"If this is unexpected, please contact the class coordinator."
        )

        if not parent_email:
            ab["skipped_no_address"] += 1
        else:
            ok, err = send_alert(parent_email, subject_line, body_text)
            if ok:
                ab["emails_sent"] += 1
            else:
                ab["smtp_failed"] += 1
                if err and len(smtp_error_samples) < 3:
                    smtp_error_samples.append(err)

        parent_phone = row.get("parent_phone")
        if parent_phone:
            ok_sms, err_sms = send_sms(parent_phone, body_text)
            if ok_sms:
                ab["sms_sent"] += 1
            else:
                ab["sms_failed"] += 1
                print("SMS sending failed:", err_sms)
                if err_sms and len(sms_error_samples) < 3:
                    sms_error_samples.append(str(err_sms))
        else:
            ab["skipped_no_phone"] += 1

        percentage = calculate_percentage(raw_id)
        if percentage < 75:
            low_subject_line = "Low Attendance Alert"
            low_body_text = (
                f"Your child {student_name}'s overall attendance has dropped to {percentage}%.\n"
                f"Minimum required attendance is 75%."
            )
            if parent_email:
                ok, err = send_alert(parent_email, low_subject_line, low_body_text)
                if not ok and err and len(smtp_error_samples) < 3:
                    smtp_error_samples.append(err)
            
            if parent_phone:
                ok_sms, err_sms = send_sms(parent_phone, low_body_text)
                if not ok_sms:
                    print("Low Attendance SMS sending failed:", err_sms)
                    if err_sms and len(sms_error_samples) < 3:
                        sms_error_samples.append(str(err_sms))

    return ab, smtp_error_samples, sms_error_samples
@app.route('/take_attendance')
def take_attendance():
    if "teacher" not in session:
        return redirect("/teacher_login")

    tid = session["teacher"]
    cursor.execute("SELECT * FROM teachers WHERE id=%s", (tid,))
    teacher_row = cursor.fetchone()
    if not teacher_row:
        session.pop("teacher", None)
        return redirect("/teacher_login")
    teacher_name = teacher_row[1]

    day = request.args.get("day")
    hour = request.args.get("hour")
    subject = request.args.get("subject")
    batch_raw = request.args.get("batch")
    batch_norm = _normalize_batch_value(batch_raw)

    if not batch_norm:
        return (
            "Batch is required for this class. Set batch on the timetable entry "
            "and ensure it matches student records, then use Take Attendance from your dashboard row."
        )

    if not day or hour is None or str(hour).strip() == "" or not subject:
        return (
            "Incomplete class link. Open the Teacher Dashboard and use "
            "Take Attendance on your class row."
        )

    try:
        hour_int = int(hour)
    except (TypeError, ValueError):
        return "Invalid period (hour) in the attendance link."

    if not _teacher_slot_allows_batch(teacher_name, day, hour_int, subject, batch_norm):
        return (
            "This class slot is not assigned to you, or the batch does not match "
            "your timetable. Use Take Attendance only from your dashboard."
        )

    batch = batch_norm

    date_today = datetime.date.today()
    date_yesterday = date_today - datetime.timedelta(days=1)
    date_str = request.args.get("date")
    if date_str:
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format in the attendance link."
    else:
        target_date = date_today

    if target_date != date_today and target_date != date_yesterday:
        return "Attendance can only be taken for the current day or the previous day."

    if is_holiday(target_date):
        return "Holiday - Attendance not required"

    # Always start fresh — clear any old draft so retaking attendance resets all results
    existing_draft = session.get("attendance_draft")
    if existing_draft:
        _clear_attendance_draft()
        existing_draft = None
    resuming = False

    if _attendance_slot_taken_in_db(subject, target_date, hour_int) and not resuming:
        return "Attendance has already been taken for this subject for this hour on this date."

    student_dicts = _students_in_batch_as_dicts(batch)
    if not student_dicts:
        return (
            f"No students found for batch {batch!r}. "
            "Make sure Students.batch matches the timetable batch (same spelling; spaces/case are OK). "
            "Then run Take Attendance again."
        )

    # Fetch all active student IDs and their normalized batches from DB to ensure only current students are recognized
    cursor.execute("SELECT id, batch FROM students")
    active_student_batches = {}
    for r in cursor.fetchall():
        nid = _norm_student_id(r[0])
        if nid:
            active_student_batches[nid] = _normalize_batch_value(r[1])

    known_encodings = list(_known_encodings_cache)
    known_norm_ids = list(_known_norm_ids_cache)
    id_to_batch = dict(_id_to_batch_cache)
    id_to_raw_id = dict(_id_to_raw_id_cache)

    student_encoding_map = _student_encoding_map(known_encodings, known_norm_ids)

    import threading
    import time

    # Use DirectShow backend on Windows — avoids slow MSMF (Media Foundation)
    # default which can introduce 1–5 s buffer lag and cause the camera to appear frozen.
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # 640×480 is optimal: fast to read, enough resolution for face detection
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # keep only the latest frame in buffer
    # Drain stale frames so we don't display buffered garbage on first open
    for _ in range(5):
        video.read()

    marked = set(existing_draft.get("marked", [])) if resuming else set()
    match_streak = {}
    active_roll = None
    active_miss_count = 0

    # Use a Queue(maxsize=1) so the worker always gets the LATEST frame without
    # blocking the main display thread (drop-if-full semantics).
    import queue as _queue
    frame_queue   = _queue.Queue(maxsize=1)
    face_names_labels = []
    thread_running    = True
    lock              = threading.Lock()

    def _draw_face_overlay(frame, top, right, bottom, left, line1, line2, color):
        """Draw box + two-line label above the face (roll no + status)."""
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thick = 2
        lines = [line1]
        if line2:
            lines.append(line2)
        y = top - 8
        for text in reversed(lines):
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            y -= th + 6
            cv2.rectangle(frame, (left, y), (left + tw + 8, y + th + 8), color, -1)
            cv2.putText(frame, text, (left + 4, y + th + 4), font, scale, (255, 255, 255), thick)

    # Background face recognition worker function
    def face_recognition_worker():
        nonlocal face_names_labels, thread_running
        nonlocal match_streak, marked, active_student_batches

        import time as _time
        fps_time    = _time.time()
        fps_counter = 0
        fps_display = 0.0
        target_batch_norm = _normalize_batch_value(batch)
        # CNN rate-limiting: only run CNN fallback at most once every 3 seconds
        last_cnn_time = 0.0
        CNN_COOLDOWN  = 3.0

        while thread_running:
            # Block until a new frame is available (timeout=0.1 s to allow clean exit)
            try:
                frame_to_process = frame_queue.get(timeout=0.1)
            except _queue.Empty:
                continue

            # ── Downscale to 40% of input for fast processing ──
            # 640×480 × 0.4 = 256×192  → very fast HOG + encoding
            INV = 2.5   # display boxes scale back up by this factor
            rgb_small = cv2.resize(frame_to_process, (0, 0), fx=0.4, fy=0.4)
            rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
            rgb_small = _enhance_rgb_for_recognition(rgb_small)

            # PRIMARY: HOG with upsample=1 — fast and handles normal webcam distances
            faces = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=1, model="hog"
            )

            # CNN FALLBACK (rate-limited): only fires at most every 3 seconds so it
            # cannot block the camera. Handles glasses, extreme angles, expressions.
            now = _time.time()
            if not faces and (now - last_cnn_time) >= CNN_COOLDOWN:
                try:
                    faces = face_recognition.face_locations(
                        rgb_small, number_of_times_to_upsample=1, model="cnn"
                    )
                    last_cnn_time = _time.time()
                except Exception:
                    faces = []

            # Scale face locations back to original frame size for drawing
            scaled_faces = [
                (int(t * INV), int(r * INV), int(b * INV), int(l * INV))
                for t, r, b, l in faces
            ]

            # Encode with jitters=1 for maximum speed.
            # Accuracy is maintained by MATCH_CONFIRM_FRAMES streak requirement.
            encodes = face_recognition.face_encodings(
                rgb_small, faces,
                num_jitters=1,
                model="large",
            )

            # FPS tracking
            fps_counter += 1
            elapsed = _time.time() - fps_time
            if elapsed >= 1.0:
                fps_display = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_time    = _time.time()

            new_face_names_labels = []
            new_face_names_labels.append(('fps', fps_display))  # special FPS token

            for encode, faceLoc in zip(encodes, scaled_faces):

                top, right, bottom, left = faceLoc

                if not student_encoding_map:
                    new_face_names_labels.append(
                        (top, right, bottom, left, "Face undetected", "", (0, 0, 255))
                    )
                    continue

                matched_id, _, _ = _match_face_to_student(
                    encode, student_encoding_map, tolerance=FACE_MATCH_TOLERANCE
                )

                # Reject if not an active DB student
                if matched_id and matched_id not in active_student_batches:
                    matched_id = None

                # ---- BEGIN CENTROID TRACKING FALLBACK ----
                if not matched_id:
                    cx = (left + right) / 2
                    cy = (top + bottom) / 2
                    best_prev = None
                    best_dist = 100000
                    with lock:
                        prev_labels = list(face_names_labels)
                    for p in prev_labels:
                        if len(p) == 7:
                            ptop, pright, pbottom, pleft, pline1, pline2, pcolor = p
                            pcx = (pleft + pright) / 2
                            pcy = (ptop + pbottom) / 2
                            dist = ((cx - pcx)**2 + (cy - pcy)**2)**0.5
                            if dist < 120 and dist < best_dist:
                                best_dist = dist
                                best_prev = p
                    if best_prev and best_prev[4].startswith("Roll:"):
                        prev_id = best_prev[4].replace("Roll: ", "")
                        if _norm_student_id(prev_id) in marked:
                            matched_id = prev_id
                # ---- END CENTROID TRACKING FALLBACK ----

                if matched_id:
                    student_batch = _id_to_batch_cache.get(matched_id)
                    if student_batch == target_batch_norm:
                        with lock:
                            match_streak[matched_id] = match_streak.get(matched_id, 0) + 1
                            streak = match_streak[matched_id]
                        if streak >= MATCH_CONFIRM_FRAMES:
                            nid = _norm_student_id(matched_id)
                            if nid:
                                marked.add(nid)

                        is_marked = (_norm_student_id(matched_id) in marked)
                        new_face_names_labels.append((
                            top, right, bottom, left,
                            f"Roll: {matched_id}",
                            "PRESENT" if is_marked else "Recognizing...",
                            (0, 255, 0),
                        ))
                    else:
                        new_face_names_labels.append((
                            top, right, bottom, left,
                            f"Roll: {matched_id}",
                            "Not in current batch",
                            (0, 0, 255),
                        ))
                else:
                    new_face_names_labels.append((
                        top, right, bottom, left,
                        "Face undetected", "", (0, 0, 255),
                    ))

            # Publish results (lock-free swap)
            with lock:
                face_names_labels = new_face_names_labels

    # Start worker thread
    worker_thread = threading.Thread(target=face_recognition_worker, daemon=True)
    worker_thread.start()

    # ── Dedicated camera capture thread ──────────────────────────────────────
    # Continuously drains the camera buffer in the background so the display
    # thread never blocks waiting for video.read().  This is the primary fix
    # for the camera appearing frozen: on Windows, MSMF/DirectShow buffers can
    # stall the caller for hundreds of milliseconds if not drained continuously.
    _latest_frame      = [None]   # list used as a mutable cell
    _capture_lock      = threading.Lock()
    _capture_running   = True

    def _camera_capture_thread():
        while _capture_running:
            ret, frm = video.read()
            if ret:
                with _capture_lock:
                    _latest_frame[0] = frm
            else:
                time.sleep(0.005)   # tiny back-off on read failure

    capture_thread = threading.Thread(target=_camera_capture_thread, daemon=True)
    capture_thread.start()

    # Main thread: overlay drawing — never blocks on camera I/O
    while True:
        with _capture_lock:
            frame = _latest_frame[0]

        if frame is None:
            # Capture thread not ready yet — pump GUI events and wait
            if cv2.waitKey(5) == 27:
                break
            time.sleep(0.005)
            continue

        frame = frame.copy()  # own copy so capture thread can update freely

        # Push latest frame to recognition worker (drop if worker still busy)
        try:
            frame_queue.put_nowait(frame.copy())
        except Exception:
            pass  # drop frame — worker busy, display keeps running smoothly

        # Render cached overlays on every display frame (no wait for worker)
        with lock:
            current_overlays = list(face_names_labels)
            marked_count     = len(marked)

        fps_val = 0.0
        for item in current_overlays:
            if item[0] == 'fps':
                fps_val = item[1]
                continue
            top, right, bottom, left, line1, line2, color = item
            _draw_face_overlay(frame, top, right, bottom, left, line1, line2, color)

        # HUD: recognition FPS (worker speed) + student count
        hud = f"FPS: {fps_val}  |  Detected: {marked_count}/{len(student_dicts)}  (Esc = review)"
        cv2.rectangle(frame, (0, 0), (380, 30), (15, 15, 15), -1)
        cv2.putText(frame, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 2)

        cv2.imshow("Attendance Camera", frame)

        if cv2.waitKey(5) == 27:   # 5 ms gives Windows GUI enough time to stay responsive
            break

    # Cleanup — stop all threads before releasing the camera
    thread_running   = False
    _capture_running = False
    capture_thread.join(timeout=1.0)
    worker_thread.join(timeout=1.0)
    video.release()
    cv2.destroyAllWindows()

    _save_attendance_draft({
        "subject": subject,
        "target_date": target_date.isoformat(),
        "hour_int": hour_int,
        "batch": batch,
        "batch_raw": batch_raw,
        "day": day,
        "marked": list(marked),
        "teacher_id": tid,
    })

    return redirect(url_for("attendance_review"))
@app.route("/attendance_review")
def attendance_review():
    if "teacher" not in session:
        return redirect("/teacher_login")

    draft = session.get("attendance_draft")
    if not draft:
        return redirect("/teacher")

    tid = session["teacher"]
    cursor.execute("SELECT * FROM teachers WHERE id=%s", (tid,))
    teacher_row = cursor.fetchone()
    if not teacher_row:
        session.pop("teacher", None)
        return redirect("/teacher_login")
    teacher_name = teacher_row[1]

    subject = draft["subject"]
    target_date = datetime.datetime.strptime(draft["target_date"], "%Y-%m-%d").date()
    hour_int = int(draft["hour_int"])
    batch = draft["batch"]
    marked_set = set(draft.get("marked", []))

    student_dicts = _students_in_batch_as_dicts(batch)
    present_students, absent_students = _student_lists_from_marked(student_dicts, marked_set)

    retake_url = url_for(
        "take_attendance",
        day=draft.get("day", ""),
        hour=hour_int,
        subject=subject,
        batch=draft.get("batch_raw", batch),
        date=draft["target_date"],
    )

    return render_template(
        "attendance_review.html",
        present_students=present_students,
        absent_students=absent_students,
        subject=subject,
        date=draft["target_date"],
        hour=hour_int,
        batch=batch,
        teacher_name=teacher_name,
        retake_url=retake_url,
        detected_count=len(present_students),
        undetected_count=len(absent_students),
    )


@app.route("/remove_from_present", methods=["POST"])
def remove_from_present():
    """Remove a student from the present (marked) list in the current draft."""
    if "teacher" not in session:
        return redirect("/teacher_login")
    draft = session.get("attendance_draft")
    if not draft:
        return redirect("/teacher")
    student_id = _norm_student_id(request.form.get("student_id", ""))
    if student_id:
        marked = set(draft.get("marked", []))
        marked.discard(student_id)
        draft["marked"] = list(marked)
        session["attendance_draft"] = draft
        session.modified = True
    return redirect(url_for("attendance_review"))

@app.route("/confirm_attendance", methods=["POST"])
def confirm_attendance():
    if "teacher" not in session:
        return redirect("/teacher_login")

    draft = session.get("attendance_draft")
    if not draft:
        return redirect("/teacher")

    tid = session["teacher"]
    if draft.get("teacher_id") != tid:
        return "Session mismatch. Please take attendance again from your dashboard."

    cursor.execute("SELECT * FROM teachers WHERE id=%s", (tid,))
    teacher_row = cursor.fetchone()
    teacher_name = teacher_row[1] if teacher_row else "Teacher"

    subject = draft["subject"]
    target_date = datetime.datetime.strptime(draft["target_date"], "%Y-%m-%d").date()
    hour_int = int(draft["hour_int"])
    batch = draft["batch"]
    marked_set = set(draft.get("marked", []))

    if _attendance_slot_taken_in_db(subject, target_date, hour_int):
        _clear_attendance_draft()
        return "Attendance was already confirmed for this class."

    student_dicts = _students_in_batch_as_dicts(batch)
    ab, smtp_error_samples, sms_error_samples = _finalize_attendance(
        student_dicts, marked_set, subject, target_date, hour_int, batch
    )
    _clear_attendance_draft()

    present_students, absent_students = _student_lists_from_marked(student_dicts, marked_set)

    summary = (
        f"Attendance confirmed. Present: {len(present_students)}, "
        f"Absent (not detected): {len(absent_students)}. "
        f"Parent absence emails: {ab['emails_sent']} sent, "
        f"{ab['skipped_no_address']} skipped (no email on file), "
        f"{ab['smtp_failed']} SMTP failures. "
        f"Parent absence SMS: {ab.get('sms_sent', 0)} sent, "
        f"{ab.get('skipped_no_phone', 0)} skipped (no phone on file), "
        f"{ab.get('sms_failed', 0)} failures."
    )
    if smtp_error_samples:
        summary += " Mail error: " + " | ".join(smtp_error_samples)[:400]
    if sms_error_samples:
        summary += " SMS error: " + " | ".join(sms_error_samples)[:400]

    return render_template(
        "attendance_results.html",
        success=len(present_students) > 0,
        confirmed=True,
        summary=summary,
        present_students=present_students,
        absent_students=absent_students,
        subject=subject,
        date=draft["target_date"],
        hour=hour_int,
        batch=batch,
        teacher_name=teacher_name,
    )
#---------------- REPORTS ----------------
@app.route('/report')
def report():
    from_date = request.args.get("from_date", "")
    to_date = request.args.get("to_date", "")

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    if not to_date:
        to_date = today_str

    if from_date:
        cursor.execute("""
            SELECT 
                a.id, 
                s.name, 
                a.subject, 
                a.date, 
                a.status, 
                a.time, 
                a.hour,
                a.student_id
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.id
            WHERE a.date >= %s AND a.date <= %s
            ORDER BY a.date DESC, a.id DESC
        """, (from_date, to_date))
    else:
        cursor.execute("""
            SELECT 
                a.id, 
                s.name, 
                a.subject, 
                a.date, 
                a.status, 
                a.time, 
                a.hour,
                a.student_id
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.id
            WHERE a.date <= %s
            ORDER BY a.date DESC, a.id DESC
        """, (to_date,))
    rows = cursor.fetchall()

    data = []
    for a_id, s_name, subject, a_date, status, a_time, hour, s_id in rows:
        data.append({
            "id": a_id,
            "student_name": s_name if s_name else s_id,
            "subject": subject,
            "date": a_date,
            "status": status,
            "time": a_time,
            "hour": hour if hour else "-"
        })

    return render_template("report.html", data=data, from_date=from_date, to_date=to_date)
#---------------- CLI UTILITIES ----------------
def cli_add_hour():
    print("Running database schema update: adding 'hour' column to attendance table...")
    try:
        cursor.execute("ALTER TABLE attendance ADD COLUMN hour VARCHAR(20)")
        db.commit()
        print("Column 'hour' successfully added to attendance table (or already exists).")
    except mysql.connector.Error as err:
        print("Error/Warning during schema update:", err)

def cli_db_check():
    print("Running database diagnostics check...")
    try:
        cursor.execute("SELECT id, day, hour, subject, teacher, semester, batch, date FROM timetable WHERE LOWER(TRIM(teacher)) = 'sindhu'")
        all_classes = cursor.fetchall()
        print("all_classes for Sindhu:", all_classes)

        cursor.execute("SELECT DISTINCT subject, hour FROM attendance WHERE date=CURDATE()")
        taken_attendance = [f"{row[0]}_{row[1]}" for row in cursor.fetchall()]
        print("taken_attendance today:", taken_attendance)

        for row in all_classes:
            sub_hr = f"{row[3]}_{row[2]}"
            print("Checking", sub_hr, "in", taken_attendance, "->", sub_hr in taken_attendance)
    except mysql.connector.Error as err:
        print("Database check failed:", err)
def cli_modify_templates():
    import glob, re
    print("Running template back-button modification utility...")
    pages = [
        'add_timetable.html',
        'admin_student_view.html',
        'admin_login.html',
        'edit_student.html',
        'edit_teacher.html',
        'edit_timetable.html',
        'report.html',
        'student_timetable.html',
        'timetable.html',
        'student_login.html',
        'teacher_login.html'
    ]

    back_btn_html = '<div style="padding:10px;text-align:left;"><a href="javascript:history.back()" style="text-decoration:none; background:#6c757d; color:white; padding:8px 15px; border-radius:5px; font-family:Arial;">⬅ Go Back</a></div>'

    for p in pages:
        path = os.path.join('templates', p)
        if not os.path.exists(path):
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Remove old back buttons
        content = re.sub(r'<a href="/admin" style="display:inline-block; margin-bottom: 15px; text-decoration:none; background:#6c757d; color:white; padding:8px 15px; border-radius:5px;">⬅ Back to Admin Dashboard</a>', '', content)
        content = re.sub(r'<a href="javascript:history.back\(\)" style="display:inline-block; margin-bottom: 15px; text-decoration:none; background:#6c757d; color:white; padding:8px 15px; border-radius:5px;">⬅ Go Back</a>', '', content)
        content = re.sub(r'<a href="/student" style="display:inline-block; margin-bottom: 15px; text-decoration:none; background:#6c757d; color:white; padding:8px 15px; border-radius:5px;">⬅ Back to Dashboard</a>', '', content)
        
        # Remove existing back links from bottom
        content = re.sub(r'<a href="/admin" class="back">⬅ Back to Admin Panel</a>', '', content)
        content = re.sub(r'<a href="/admin" class="btn">Back to Admin Dashboard</a>', '', content)
        content = re.sub(r'<a href="/timetable">Cancel</a>', '', content)
        content = re.sub(r'<a href="/" class="back">⬅ Back to Home</a>', '', content)
        
        # Insert new back button right after <body>
        if '⬅ Go Back' not in content:
            content = re.sub(r'(<body[^>]*>)', r'\1\n' + back_btn_html, content, count=1)
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    print("Updated all page templates with universal Back button.")
def cli_theme_update():
    import glob, re
    print("Running theme update utility across templates...")
    templates_dir = "templates"
    html_files = glob.glob(os.path.join(templates_dir, "*.html"))

    replacements = [
        # Backgrounds
        (r"background:\s*linear-gradient\(120deg,#4facfe,#00f2fe\);", "background: linear-gradient(135deg, #0f172a, #1e1b4b); color: #f8fafc;"),
        (r"background:\s*#f4f4f4;?", "background: #0f172a; color: #f8fafc;"),
        (r"background:\s*#f5f5f5;?", "background: #0f172a; color: #f8fafc;"),
        (r"background:\s*#eef2f7;?", "background: #0f172a; color: #f8fafc;"),
        (r"background:\s*white;?", "background: rgba(255, 255, 255, 0.05); color: #f8fafc; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);"),
        
        # Text colors
        (r"color:\s*#2c3e50;?", "color: #f8fafc;"),
        (r"color:\s*#333;?", "color: #f8fafc;"),
        (r"color:\s*black;?", "color: #f8fafc;"),

        # Buttons and Headers
        (r"background:\s*#3498db;?", "background: linear-gradient(135deg, #8b5cf6, #3b82f6); border: none;"),
        (r"background:\s*#4facfe;?", "background: linear-gradient(135deg, #8b5cf6, #3b82f6); border: none;"),
        (r"background:\s*#0077ff;?", "background: linear-gradient(135deg, #7c3aed, #2563eb); box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4); transform: translateY(-2px);"),
        (r"background:\s*#2ecc71;?", "background: linear-gradient(135deg, #10b981, #059669); border: none;"),
        (r"background:\s*#27ae60;?", "background: linear-gradient(135deg, #059669, #047857); box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); transform: translateY(-2px);"),

        # Borders and Shadows
        (r"border:\s*1px solid #ccc;?", "border: 1px solid rgba(255, 255, 255, 0.1);"),
        (r"border:\s*1px solid #ddd;?", "border: 1px solid rgba(255, 255, 255, 0.1);"),
        (r"box-shadow:\s*0 0 10px gray;?", "box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);"),
        (r"box-shadow:\s*0px 0px 10px #ccc;?", "box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);"),

        # Table text colors
        (r"th\s*{\s*background:[^;]+;\s*color:\s*white;?", "th {\nbackground: rgba(255,255,255,0.1);\ncolor: #f8fafc;\nbackdrop-filter: blur(5px);"),
        
        # Specific elements
        (r"font-family:\s*Arial;", "font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;"),
        (r"a:hover\s*{", "a { transition: all 0.3s ease; }\n\na:hover{"),
        (r"button:hover\s*{", "button { transition: all 0.3s ease; }\n\nbutton:hover{"),
    ]

    for file_path in html_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
            
        # Inject universal CSS for inputs/buttons if not present
        if "<style>" in content and "input[type" not in content:
            extra_css = """
input, select {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 5px;
    outline: none;
}
input, select {
    border-color: #8b5cf6;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.5);
}
button, .btn {
    transition: all 0.3s ease;
    cursor: pointer;
}
"""
            content = content.replace("<style>", "<style>\n" + extra_css)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {file_path}")
    print("Theme update finished.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--add-hour":
            cli_add_hour()
        elif arg == "--db-check":
            cli_db_check()
        elif arg == "--modify-templates":
            cli_modify_templates()
        elif arg == "--theme-update":
            cli_theme_update()
        elif arg in ("--help", "-h"):
            print("Usage: python attendance.py [options]")
            print("Options:")
            print("  --add-hour           Run the database schema alteration to add 'hour' column to the attendance table")
            print("  --db-check           Check database records for classes, attendance today, etc.")
            print("  --modify-templates   Modify HTML templates with modern Back Button link structure")
            print("  --theme-update       Perform search-and-replace style theme overrides across template directories")
            print("  --help, -h           Show this help message")
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help to see available options.")
    else:
        app.run(debug=True)




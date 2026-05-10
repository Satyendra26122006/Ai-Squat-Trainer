import os
import sqlite3
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, g

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "squat_trainer.db")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DATABASE"] = DB_PATH


def get_db():
    db = getattr(g, "db", None)
    if db is None:
        db = sqlite3.connect(app.config["DATABASE"])
        db.row_factory = sqlite3.Row
        g.db = db
    return db


def init_db():
    if not os.path.exists(app.config["DATABASE"]):
        conn = sqlite3.connect(app.config["DATABASE"])
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                total_reps INTEGER NOT NULL,
                avg_score REAL NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE reps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                phase TEXT NOT NULL,
                left_knee REAL,
                right_knee REAL,
                left_hip REAL,
                right_hip REAL,
                left_ankle REAL,
                right_ankle REAL,
                left_back REAL,
                right_back REAL,
                knee_over_toe REAL,
                torso_lean REAL,
                neck_angle REAL,
                knee_width_ratio REAL,
                hip_level_diff REAL,
                shoulder_level REAL,
                errors TEXT,
                score INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        conn.commit()
        conn.close()


@app.teardown_appcontext
def close_db(error):
    db = getattr(g, "db", None)
    if db is not None:
        db.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/history")
def get_history():
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, name, started_at, total_reps, avg_score, created_at FROM sessions ORDER BY started_at DESC LIMIT 20"
    )
    sessions = [dict(row) for row in cur.fetchall()]
    for session in sessions:
        session["started_at"] = datetime.fromtimestamp(session["started_at"]).isoformat()
        session["created_at"] = datetime.fromtimestamp(session["created_at"]).isoformat()
    return jsonify(sessions=sessions)


@app.route("/api/session/<int:session_id>")
def session_detail(session_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, name, started_at, total_reps, avg_score, created_at FROM sessions WHERE id = ?",
        (session_id,),
    )
    session = cur.fetchone()
    if session is None:
        return jsonify(error="Session not found"), 404

    cur.execute(
        "SELECT timestamp, phase, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle, left_back, right_back, knee_over_toe, torso_lean, neck_angle, knee_width_ratio, hip_level_diff, shoulder_level, errors, score FROM reps WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    )
    reps = [dict(row) for row in cur.fetchall()]
    for rep in reps:
        rep["timestamp"] = datetime.fromtimestamp(rep["timestamp"]).isoformat()
    result = dict(session)
    result["started_at"] = datetime.fromtimestamp(result["started_at"]).isoformat()
    result["created_at"] = datetime.fromtimestamp(result["created_at"]).isoformat()
    result["reps"] = reps
    return jsonify(result)


@app.route("/api/delete_session/<int:session_id>", methods=["DELETE", "POST"])
def delete_session(session_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM reps WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    if cur.rowcount == 0:
        return jsonify(error="Session not found"), 404
    db.commit()
    return jsonify(success=True)


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM reps")
    cur.execute("DELETE FROM sessions")
    db.commit()
    return jsonify(success=True)


@app.route("/api/save_session", methods=["POST"])
def save_session():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify(error="Invalid JSON payload"), 400

    name = payload.get("name", "AI Squat Session")
    started_at = int(payload.get("started_at", time.time()))
    total_reps = int(payload.get("total_reps", 0))
    avg_score = float(payload.get("avg_score", 0))
    reps = payload.get("reps", [])

    if total_reps == 0 or not reps:
        return jsonify(error="No reps to save"), 400

    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO sessions (name, started_at, total_reps, avg_score, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, started_at, total_reps, avg_score, int(time.time())),
    )
    session_id = cur.lastrowid

    for rep in reps:
        cur.execute(
            "INSERT INTO reps (session_id, timestamp, phase, left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle, left_back, right_back, knee_over_toe, torso_lean, neck_angle, knee_width_ratio, hip_level_diff, shoulder_level, errors, score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                int(rep.get("timestamp", time.time())),
                rep.get("phase", "UNKNOWN"),
                rep.get("left_knee"),
                rep.get("right_knee"),
                rep.get("left_hip"),
                rep.get("right_hip"),
                rep.get("left_ankle"),
                rep.get("right_ankle"),
                rep.get("left_back"),
                rep.get("right_back"),
                rep.get("knee_over_toe"),
                rep.get("torso_lean"),
                rep.get("neck_angle"),
                rep.get("knee_width_ratio"),
                rep.get("hip_level_diff"),
                rep.get("shoulder_level"),
                ",".join(rep.get("errors", [])),
                int(rep.get("score", 0)),
            ),
        )

    db.commit()
    return jsonify(success=True, session_id=session_id)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)

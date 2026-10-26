import os
import uuid

from flask import Flask, flash, redirect, render_template, request, url_for

from netron_wrapper import start_netron
from parsers import SUPPORTED_EXTENSIONS, analyze_model

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "dev-only-local-tool"  # local single-user tool, no real auth needed
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB, model files can be large


@app.route("/")
def index():
    return render_template("index.html", extensions=SUPPORTED_EXTENSIONS)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("model_file")
    if not file or file.filename == "":
        flash("Model file တစ်ခုကို ရွေးချယ်ပေးပါ / Please choose a model file.")
        return redirect(url_for("index"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        flash(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        return redirect(url_for("index"))

    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)

    try:
        result = analyze_model(save_path)
    except Exception as exc:
        flash(f"Failed to analyze model: {exc}")
        return redirect(url_for("index"))

    try:
        netron_url = start_netron(save_path)
    except Exception:
        netron_url = None  # Detailed view is a bonus; simplified view still works without it.

    return render_template(
        "result.html",
        filename=file.filename,
        result=result,
        netron_url=netron_url,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

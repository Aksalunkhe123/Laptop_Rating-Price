

from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import os
import re

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "laptop-analyzer-local-key")

# Load trained model and dataset
model = joblib.load("laptop_price_pipeline.pkl")
df = pd.read_csv("laptop_recommendation_data.csv")


def normalize_text(value):
    """Normalize internal whitespace and strip trailing spaces."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


# Extract encoder categories to preserve exact ML model feature compatibility
cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
_trained_brands = list(cat_encoder.categories_[0])
_trained_processors = list(cat_encoder.categories_[1])
_trained_os = list(cat_encoder.categories_[2])

# Map normalized dropdown selections back to exact trained feature strings
_brand_model_map = {normalize_text(b): b for b in _trained_brands}
_processor_model_map = {normalize_text(p): p for p in _trained_processors}
_os_model_map = {normalize_text(o): o for o in _trained_os}

# Prepare clean columns in dataset for deduplication, matching, and display
df["clean_brand"] = df["brand"].apply(normalize_text)
df["clean_processor"] = df["processor"].apply(normalize_text)
df["clean_os"] = df["OS"].apply(normalize_text)
df["clean_ram"] = pd.to_numeric(df["Ram"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def clean_value(value, default="Not available"):
    """Convert pandas/numpy values into safe, readable Python values."""
    if value is None or pd.isna(value):
        return default
    if hasattr(value, "item"):
        value = value.item()
    return value


def get_column_value(row, possible_names, default="Not available"):
    """Find a column even if the dataset uses a slightly different name."""
    for column in possible_names:
        if column in row.index:
            return clean_value(row[column], default)
    return default


def get_numeric(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_laptop_details(row):
    """Convert a dataset row into a template-friendly dictionary."""
    ram_raw = get_column_value(row, ["clean_ram", "Ram", "RAM"])
    try:
        ram_display = f"{int(float(ram_raw))} GB"
    except (ValueError, TypeError):
        ram_display = str(ram_raw)

    res_w = get_column_value(row, ["resolution_width"])
    res_h = get_column_value(row, ["resolution_height"])
    try:
        res_w_display = str(int(float(res_w)))
    except (ValueError, TypeError):
        res_w_display = str(res_w)
    try:
        res_h_display = str(int(float(res_h)))
    except (ValueError, TypeError):
        res_h_display = str(res_h)

    proc_display = normalize_text(get_column_value(row, ["clean_processor", "processor", "CPU"]))
    os_display = normalize_text(get_column_value(row, ["clean_os", "OS", "os"]))
    brand_display = normalize_text(get_column_value(row, ["clean_brand", "brand"]))

    return {
        "name": str(get_column_value(row, ["name", "model"], "Laptop")),
        "brand": brand_display,
        "price": get_numeric(get_column_value(row, ["price"], 0)),
        "rating": get_numeric(
            get_column_value(row, ["rating_out_of_5", "spec_rating", "rating_5"], 0)
        ),
        "processor": proc_display,
        "cpu": str(get_column_value(row, ["CPU", "processor"])),
        "ram": ram_display,
        "ram_type": str(get_column_value(row, ["Ram_type", "RAM_type"])),
        "storage": str(get_column_value(row, ["ROM", "storage"])),
        "storage_type": str(get_column_value(row, ["ROM_type", "storage_type"])),
        "gpu": str(get_column_value(row, ["GPU"])),
        "display_size": str(get_column_value(row, ["display_size"])),
        "resolution_width": res_w_display,
        "resolution_height": res_h_display,
        "os": os_display,
        "warranty": str(get_column_value(row, ["warranty"]))
    }


def recommend_uses(laptop):
    """Generate basic suitability suggestions from available specifications."""
    ram_text = str(laptop["ram"]).lower()
    gpu_text = str(laptop["gpu"]).lower()
    processor_text = str(laptop["processor"]).lower()

    # Extract RAM size, e.g. "16 GB" -> 16
    ram_match = re.search(r"\d+", ram_text)
    ram_gb = int(ram_match.group()) if ram_match else 0

    dedicated_gpu = any(
        keyword in gpu_text
        for keyword in ["nvidia", "geforce", "rtx", "gtx", "radeon rx"]
    )
    high_performance_cpu = any(
        keyword in processor_text
        for keyword in ["i7", "i9", "ryzen 7", "ryzen 9", "ultra 7", "ultra 9"]
    )

    recommended = [
        "Everyday computing",
        "Browsing and online classes"
    ]

    not_recommended = []

    if ram_gb >= 8:
        recommended.extend(["Programming", "Web Development"])
    else:
        not_recommended.extend(["Heavy multitasking", "Large development projects"])

    if ram_gb >= 16:
        recommended.extend(["Data Science", "Machine Learning coursework"])
    else:
        not_recommended.append("Large Machine Learning workloads")

    if dedicated_gpu:
        recommended.extend(["Gaming", "GPU-accelerated applications"])
    else:
        not_recommended.append("High-end gaming")

    if dedicated_gpu and ram_gb >= 16 and high_performance_cpu:
        recommended.append("Video Editing")
    else:
        not_recommended.append("Professional 4K video editing")

    # Remove duplicate suggestions while preserving order
    recommended = list(dict.fromkeys(recommended))
    not_recommended = list(dict.fromkeys(not_recommended))

    return recommended, not_recommended


@app.route("/")
def home():
    brands = sorted(df["clean_brand"].dropna().unique()) if "clean_brand" in df else []
    processors = sorted(
        df["clean_processor"].dropna().unique()
    ) if "clean_processor" in df else []
    ram_list = sorted(
        [int(r) for r in df["clean_ram"].dropna().unique()]
    ) if "clean_ram" in df else []
    os_list = sorted(df["clean_os"].dropna().unique()) if "clean_os" in df else []

    return render_template(
        "index.html",
        brands=brands,
        processors=processors,
        ram_list=ram_list,
        os_list=os_list
    )


@app.route("/predict", methods=["POST"])
def predict():
    brand = request.form.get("brand", "").strip()
    processor = request.form.get("processor", "").strip()
    ram_raw = request.form.get("ram", "").strip()
    operating_system = request.form.get("os", "").strip()

    if not all([brand, processor, ram_raw, operating_system]):
        return "Please select Brand, Processor, RAM and Operating System.", 400

    # Extract numeric RAM value
    ram_match = re.search(r"\d+", ram_raw)
    ram_val = float(ram_match.group()) if ram_match else 8.0

    norm_brand = normalize_text(brand)
    norm_proc = normalize_text(processor)
    norm_os = normalize_text(operating_system)

    # Map normalized inputs to trained feature values to preserve exact ML model weights
    model_brand = _brand_model_map.get(norm_brand, brand)
    model_processor = _processor_model_map.get(norm_proc, processor)
    model_os = _os_model_map.get(norm_os, operating_system)

    # These feature names and types match the pipeline schema
    input_data = pd.DataFrame([{
        "brand": model_brand,
        "processor": model_processor,
        "Ram": ram_val,
        "OS": model_os
    }])

    try:
        predicted_price = float(model.predict(input_data)[0])
    except Exception as error:
        app.logger.exception("Price prediction failed")
        return (
            "Prediction failed. Check that the model was trained using "
            "the features brand, processor, Ram and OS. "
            f"Details: {error}",
            500
        )

    # Find the closest matching dataset row using clean normalized fields
    matches = df.copy()

    filters = [
        ("clean_brand", norm_brand),
        ("clean_processor", norm_proc),
        ("clean_ram", ram_val),
        ("clean_os", norm_os)
    ]

    for column, selected_value in filters:
        if column in matches.columns:
            matches = matches[matches[column] == selected_value]

    # If the exact combination isn't present, progressively relax the match
    if matches.empty:
        matches = df.copy()
        if "clean_brand" in matches.columns and "clean_ram" in matches.columns:
            matches = matches[
                (matches["clean_brand"] == norm_brand) &
                (matches["clean_ram"] == ram_val)
            ]

    if matches.empty and "clean_brand" in df.columns:
        matches = df[df["clean_brand"] == norm_brand]

    if matches.empty:
        matches = df.copy()

    # Select the closest-priced row from the remaining candidates
    if "price" in matches.columns:
        price_values = pd.to_numeric(matches["price"], errors="coerce")
        valid_prices = price_values.notna()
        if valid_prices.any():
            matches = matches.loc[valid_prices].copy()
            matches["_price_difference"] = (
                pd.to_numeric(matches["price"], errors="coerce") - predicted_price
            ).abs()
            best_row = matches.sort_values("_price_difference").iloc[0]
        else:
            best_row = matches.iloc[0]
    else:
        best_row = matches.iloc[0]

    laptop = build_laptop_details(best_row)
    recommended, not_recommended = recommend_uses(laptop)

    verdict = (
        "Excellent Specifications" if laptop["rating"] >= 4.0
        else "Good Specifications" if laptop["rating"] >= 3.0
        else "Basic Specifications"
    )

    # Prepare similar laptops, excluding the selected row where possible
    similar = df.copy()
    if "clean_brand" in similar.columns:
        similar = similar[similar["clean_brand"] == norm_brand]

    if "name" in similar.columns and laptop["name"] != "Not available":
        similar = similar[similar["name"].astype(str) != laptop["name"]]

    if "price" in similar.columns:
        similar["_price_difference"] = (
            pd.to_numeric(similar["price"], errors="coerce") - predicted_price
        ).abs()
        similar = similar.sort_values("_price_difference")

    similar_laptops = [
        build_laptop_details(row)
        for _, row in similar.head(6).iterrows()
    ]

    # Store template-ready data for navigation to the other pages
    session["analysis"] = {
        "predicted_price": round(predicted_price, 2),
        "laptop": laptop,
        "verdict": verdict,
        "recommended": recommended,
        "not_recommended": not_recommended,
        "similar_laptops": similar_laptops
    }

    return redirect(url_for("result"))



@app.route("/result")
def result():
    analysis = session.get("analysis")

    if not analysis:
        return redirect(url_for("home"))

    return render_template(
        "result.html",
        predicted_price=analysis["predicted_price"],
        laptop=analysis["laptop"],
        verdict=analysis["verdict"],
        recommended=analysis["recommended"],
        not_recommended=analysis["not_recommended"],
        similar_laptops=analysis["similar_laptops"]
    )



@app.route("/specifications")
def specifications():
    analysis = session.get("analysis")
    if not analysis:
        return redirect(url_for("home"))
    return render_template(
        "specifications.html",
        laptop=analysis["laptop"]
    )


@app.route("/recommendations")
def recommendations():
    analysis = session.get("analysis")
    if not analysis:
        return redirect(url_for("home"))
    return render_template(
        "recommendations.html",
        recommended=analysis["recommended"],
        not_recommended=analysis["not_recommended"]
    )


@app.route("/similar-laptops")
def similar_laptops():
    analysis = session.get("analysis")
    if not analysis:
        return redirect(url_for("home"))
    return render_template(
        "similar_laptops.html",
        similar_laptops=analysis["similar_laptops"]
    )


if __name__ == "__main__":
    app.run(debug=True)
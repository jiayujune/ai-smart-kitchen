from flask import Flask, render_template, request
from recommender import SmartKitchenRecommender

app = Flask(__name__)
recommender = SmartKitchenRecommender("recipes_clean.json")


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("ingredients", "")
        user_items = [item.strip() for item in user_input.split(",") if item.strip()]

        results = recommender.recommend_recipes(
            user_items,
            top_k=10,
            min_score=0.15,
            min_matched_count=2,
            max_missing_count=6
        )

    return render_template("index.html", results=results, user_input=user_input)


if __name__ == "__main__":
    app.run(debug=True)
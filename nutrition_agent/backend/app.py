from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

IBM_API_KEY = "TztZRuuH2XI6DIld6vUPdyvgaqosAJpn5Sfhnd0X7ij0"
IBM_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
MODEL_ID = "ibm/granite-4-h-small"
PROJECT_ID = "80b9bb81-01cd-4dcb-9d2a-b6bcbf65d08f"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_iam_token():
    r = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": IBM_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def call_granite(prompt, max_tokens=1500, min_tokens=100):
    token = get_iam_token()
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "repetition_penalty": 1.1,
        },
    }
    r = requests.post(
        IBM_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["results"][0]["generated_text"]


def profile_summary(p):
    """Build a compact one-line profile string from a profile dict."""
    return (
        f"Age {p.get('age','?')}, {p.get('gender','?')}, "
        f"{p.get('weight','?')}kg, {p.get('height','?')}cm, "
        f"goal: {p.get('goal','general health')}, "
        f"activity: {p.get('activity_level','moderate')}, "
        f"diet: {p.get('dietary_preference','no restrictions')}, "
        f"allergies: {p.get('allergies','none')}, "
        f"conditions: {p.get('health_conditions','none')}"
    )


def api_error(e):
    msg = f"IBM API error: {e}" if isinstance(e, requests.exceptions.HTTPError) else str(e)
    return jsonify({"error": msg}), 500


# ── Existing endpoint: full diet plan ─────────────────────────────────────────

def build_nutrition_prompt(user_data):
    p = user_data
    prompt = f"""You are a certified nutritionist and dietitian AI assistant. Provide personalized, science-based meal and diet recommendations.

User Profile: {profile_summary(p)}
Meals per day: {p.get('meals_per_day', '3')}, Cuisine: {p.get('cuisine_preference', 'any')}

Please provide:
1. **Daily Caloric Needs** - Estimated calories
2. **Macronutrient Breakdown** - Protein, carbs, fats in grams and percentages
3. **Full Day Meal Plan** - Breakfast, lunch, dinner, snacks with portions
4. **Top 10 Recommended Foods** for this user's goals
5. **Foods to Avoid**
6. **Hydration Recommendation**
7. **Key Nutritional Tips** (3-5 actionable)
8. **Supplement Suggestions** (remind to consult a doctor)

Format with clear headings and bullet points. Be specific, practical, and motivating."""
    return prompt


@app.route("/api/nutrition", methods=["POST"])
def get_nutrition_plan():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        text = call_granite(build_nutrition_prompt(data), max_tokens=1500, min_tokens=200)
        return jsonify({"plan": text, "status": "success"})
    except Exception as e:
        return api_error(e)


# ── NEW: Chat assistant ────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json()
        message = (body.get("message") or "").strip()
        profile = body.get("profile", {})
        if not message:
            return jsonify({"error": "Empty message"}), 400

        profile_ctx = f"\nUser profile: {profile_summary(profile)}" if any(profile.values()) else ""
        prompt = f"""You are NutriBot, a friendly and knowledgeable AI nutrition assistant. Answer only questions related to nutrition, food, diet, calories, healthy eating, meal planning, and wellness.{profile_ctx}

User question: {message}

Provide a helpful, accurate, and concise answer. Use bullet points where appropriate. If the question is not about nutrition or food, politely say you can only help with nutrition topics."""

        text = call_granite(prompt, max_tokens=600, min_tokens=50)
        return jsonify({"reply": text.strip(), "status": "success"})
    except Exception as e:
        return api_error(e)


# ── NEW: Food analyzer ────────────────────────────────────────────────────────

@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():
    try:
        body = request.get_json()
        food = (body.get("food") or "").strip()
        portion = (body.get("portion") or "100g").strip()
        if not food:
            return jsonify({"error": "Food name required"}), 400

        prompt = f"""You are a nutrition database expert. Analyze the following food item and provide detailed nutritional information.

Food: {food}
Portion size: {portion}

Provide:
1. **Calories** - Total calories for this portion
2. **Macronutrients** - Protein (g), Carbohydrates (g), Fats (g), Fiber (g), Sugar (g)
3. **Key Vitamins & Minerals** - List the most notable ones with approximate amounts
4. **Health Benefits** - 3-4 key health benefits of this food
5. **Health Considerations** - Any cautions or who should limit this food
6. **Healthier Alternatives** - 2-3 similar foods with better nutritional profiles if applicable
7. **Best Time to Eat** - Optimal time of day for this food

Be specific with numbers. If exact values are unknown, provide reasonable estimates based on nutritional science."""

        text = call_granite(prompt, max_tokens=800, min_tokens=100)
        return jsonify({"analysis": text.strip(), "status": "success"})
    except Exception as e:
        return api_error(e)


# ── NEW: Weekly diet plan ─────────────────────────────────────────────────────

@app.route("/api/weekly-plan", methods=["POST"])
def weekly_plan():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        prompt = f"""You are a certified nutritionist. Create a complete, practical 7-day meal plan.

User Profile: {profile_summary(data)}
Meals per day: {data.get('meals_per_day', '3')}, Cuisine: {data.get('cuisine_preference', 'any')}

Produce a full 7-day plan. Use this exact format for EVERY day:

**Day 1 - Monday**
- Breakfast: [meal name] (~X kcal)
- Lunch: [meal name] (~X kcal)
- Dinner: [meal name] (~X kcal)
- Snack: [snack name] (~X kcal)
- Daily Total: ~XXXX kcal

**Day 2 - Tuesday**
...and so on through Day 7 - Sunday.

Keep meals varied and aligned with the user profile. Do not skip any day. Start immediately with Day 1."""

        text = call_granite(prompt, max_tokens=1800, min_tokens=200)
        return jsonify({"plan": text.strip(), "status": "success"})
    except Exception as e:
        return api_error(e)


# ── NEW: Health advisor ───────────────────────────────────────────────────────

@app.route("/api/health-advice", methods=["POST"])
def health_advice():
    try:
        body = request.get_json()
        topic = (body.get("topic") or "general health").strip()
        profile = body.get("profile", {})

        profile_ctx = f"\nUser profile: {profile_summary(profile)}" if any(profile.values()) else ""
        prompt = f"""You are a preventive nutrition advisor. Provide evidence-based nutritional guidance.{profile_ctx}

Topic: {topic}

Provide:
1. **Overview** - Brief explanation of why nutrition matters for this topic
2. **Key Foods to Include** - 5-7 foods with reasons
3. **Foods to Limit or Avoid** - 4-5 items with explanations
4. **Daily Nutrition Targets** - Key nutrients to focus on
5. **Practical Tips** - 4-5 actionable daily habits
6. **Sample Day Meal Ideas** - Light examples fitting this goal

End with: "DISCLAIMER: This information is for educational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider before making significant dietary changes."

Be informative but accessible."""

        text = call_granite(prompt, max_tokens=900, min_tokens=150)
        return jsonify({"advice": text.strip(), "status": "success"})
    except Exception as e:
        return api_error(e)


# ── NEW: Food recommendations ─────────────────────────────────────────────────

@app.route("/api/food-recommendations", methods=["POST"])
def food_recommendations():
    try:
        body = request.get_json()
        profile = body.get("profile", {})
        foods = body.get("foods", [])

        foods_str = ", ".join(foods) if foods else "common processed foods"
        prompt = f"""You are a nutrition expert specializing in healthy food substitutions.

User profile: {profile_summary(profile)}
Foods the user commonly eats: {foods_str}

For each food listed (or for 6 common unhealthy foods if none listed), provide:
- **[Food Name]**: Why it may not align with the user's goals
  - Healthier Alternative 1: [name] - brief reason + estimated calories saved
  - Healthier Alternative 2: [name] - brief reason + estimated calories saved
  - Preparation tip to make the original healthier

Also add a section **Top 8 Superfoods for This User's Goals** with a one-line reason for each.

Be practical and encouraging, not judgmental."""

        text = call_granite(prompt, max_tokens=900, min_tokens=150)
        return jsonify({"recommendations": text.strip(), "status": "success"})
    except Exception as e:
        return api_error(e)


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Nutrition Agent API is running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

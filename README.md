# What's For Dinner? 🍽️

A CrewAI-powered home app to help you decide what to cook for dinner, based on your preferences, dietary needs, and available ingredients.

## Features

- **Preference Analyst:** Interprets and summarizes your dietary needs and meal constraints.
- **Recipe Brainstormer:** Suggests creative and diverse dinner ideas.
- **Meal Planner:** Selects the best recipe and provides a simple cooking plan with ingredients and instructions.

## Project Structure
- `src/` — Main logic and orchestration 

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (`ollama/llama3` model)
- Install dependencies:
  ```bash
  pip install crewai litellm python-dotenv
  ```

### Setup

1. Clone the repository.
2. Ensure Ollama is running:  
   ```bash
   ollama serve
   ollama pull llama3
   ```
3. Copy `.env.example` to `.env` and configure as needed.

### Running the App

```bash
python src/dinner_recommender/main.py
```

Answer the prompts about your dinner preferences. The app will analyze your input, brainstorm recipes, and provide a detailed dinner suggestion.

## Example Output

```
Hello! I'm your AI Dinner Decider. Let's plan your meal.
Please answer the following questions about your dinner preferences:
Dietary preference (e.g., Vegetarian, Vegan, Gluten-Free, None): Vegetarian
Cuisine preference (e.g., Italian, Mexican, Asian, Indian, None): Italian
Desired cooking time (e.g., <30min, 30-60min, >60min): <30min
Ingredients to use (comma-separated, e.g., pasta, spinach): pasta, spinach
Ingredients to avoid (comma-separated, e.g., mushrooms): mushrooms

Starting the dinner planning crew...

## Here is your dinner suggestion for today!
Meal Suggestion: Speedy Spinach & Tomato Pasta
Ingredients:
- Pasta (400g)
- Fresh spinach (150g)
- Cherry tomatoes (200g)
- Garlic (2 cloves)
- Olive oil (2 tbsp)
Instructions:
1. Cook pasta according to package instructions.
2. Sauté garlic in olive oil, add spinach and tomatoes.
3. Toss cooked pasta with veggies, season, and serve.
```



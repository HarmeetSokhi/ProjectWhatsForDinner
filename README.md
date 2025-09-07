# What's For Dinner? 🍽️

A CrewAI-powered home app to help you decide what to cook for dinner, based on your preferences, dietary needs, and available ingredients.

## Features

- **Input Parser:** Corrects spelling and interprets ambiguous user input using NLP.
- **Recipe Brainstormer:** Suggests creative and diverse dinner ideas, avoiding recent repeats.
- **Meal Planner:** Selects the best recipe and provides a simple cooking plan with ingredients and instructions.
 

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (`ollama/llama3` or `llama4`)
- Install dependencies:
  ```bash
  pip install crewai litellm python-dotenv streamlit
  ```

### Setup

1. Clone the repository.
2. Ensure Ollama is running:  
   ```bash
   ollama serve
   ollama pull llama3
   ```
   *(Or use `llama4` if configured in your code.)*
3. Copy `.env.example` to `.env` and configure as needed.

### Running the App

#### Streamlit UI

```bash
streamlit run src/dinner_recommender/main.py
```

Fill in your dinner preferences in the web form. The app will analyze your input, brainstorm recipes (avoiding recent repeats), and provide a detailed dinner suggestion.

#### CLI (if enabled)

```bash
python src/dinner_recommender/main.py
```

## Example Output

```
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





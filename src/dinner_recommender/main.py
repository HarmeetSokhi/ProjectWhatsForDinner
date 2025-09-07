# Main entry point for dinner recommender
# Warning control
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from litellm import completion
import streamlit as st


class LiteLLMOllama:
    def __init__(self, model, base_url, temperature=0.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def __call__(self, prompt, **kwargs):
        # Ensure temperature is set unless overridden
        if 'temperature' not in kwargs:
            kwargs['temperature'] = self.temperature
        return completion(
            model=self.model,
            api_base=self.base_url,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )["choices"][0]["message"]["content"]

load_dotenv() 
# Initialize Ollama LLM (make sure Ollama server is running)
llm = LiteLLMOllama(model="ollama/llama3", base_url="http://localhost:11434", temperature=0.3)


#

# Agent 0: Input Parser and Corrector
# Role: Parses and corrects user input using NLP expertise.
# Goal: To produce a corrected, structured Python list of dictionaries from raw user input.
# Backstory: An expert in natural language processing, spelling correction, and semantic interpretation, able to infer the most meaningful words and place them in the correct dictionary fields even if the spelling is incorrect or ambiguous.
input_parser = Agent(
    role='Input Parser and NLP Corrector',
    goal='Parse raw user input, correct spelling mistakes, interpret ambiguous or misspelled words, and produce a structured Python list of dictionaries with keys: dietary, cuisine, time, ingredients_to_use, ingredients_to_avoid.',
    backstory=(
        "You are a world-class natural language processing expert and data parser. "
        "You specialize in correcting spelling mistakes, interpreting ambiguous or misspelled words, and mapping them to the most meaningful and contextually appropriate dictionary keys. "
        "You use advanced NLP techniques to ensure every piece of user input is accurately understood and placed in the correct field, even if the original input is unclear or contains errors."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 2: Recipe Brainstormer
# Role: Generates creative and diverse recipe ideas.
# Goal: To propose several meal options adhering to the analyzed preferences.
# Backstory: A seasoned chef with a global culinary background, known for innovative and delicious recipe concepts.


def read_meal_names():
    file_path = "final_meal_name.txt"
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

recipe_brainstormer = Agent(
    role='Creative Recipe Brainstormer',
    goal='Always generate 3-5 diverse and appealing meal ideas based on the analyzed preferences, interpreting the previous task\'s output.',
    backstory=(
        "You are a celebrated chef and culinary innovator, having traveled the world to master flavors, techniques, and traditions from every continent. "
        "Your creativity knows no bounds—you effortlessly blend classic recipes with modern twists, and always find ways to make meals exciting and accessible. "
        "You thrive on variety and inclusivity, ensuring your ideas suit any dietary need or ingredient constraint. "
        "Whether the input is clear or vague, you can conjure up delicious, unique, and practical dinner ideas that inspire home cooks and delight every palate."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 3: Meal Planner and Instruction Provider
# Role: Selects the best recipe and creates a simple, executable cooking plan.
# Goal: To deliver one clear, actionable dinner suggestion with ingredients and instructions.
# Backstory: A highly organized and practical meal prep specialist, making complex recipes simple for home cooks.
meal_planner = Agent(
    role='Daily Meal Planner and Instruction Provider',
    goal='Always select and detail a dinner meal suggestion, providing a concise cooking plan and ingredient list for a single dinner, even if recipe ideas or preferences are incomplete, minimal, or unclear.',
    backstory=(
        "You are a master meal planner and home cooking coach, renowned for your ability to turn any set of ingredients and preferences into a delicious, stress-free dinner. "
        "With years of experience organizing family kitchens and teaching busy people how to cook, you excel at simplifying complex recipes and adapting instructions for all skill levels. "
        "Your approach is practical, empathetic, and detail-oriented, ensuring every meal is easy to follow, nutritious, and enjoyable. "
        "You always anticipate potential challenges and provide clear, step-by-step guidance, making home cooking accessible and rewarding for everyone."
    ),
    verbose=True,
    allow_delegation=False, # This agent delivers the final output and does not delegate its primary task
    llm=llm # Assign the configured LLM
)



# --- Define Tasks ---

# Task 0: Parse and correct user input
parse_and_correct_input_task = Task(
    description=(
        "You are given five separate strings: {dietary}, {cuisine}, {time}, {ingredients_to_use}, {ingredients_to_avoid}. "
        "Parse each value, correct spelling mistakes, interpret ambiguous or misspelled words, and generate a valid Python list of one dictionary with keys: 'dietary', 'cuisine', 'time', 'ingredients_to_use', 'ingredients_to_avoid'. "
        "Each key should have the corrected and most meaningful value. If any value is unclear or missing, infer a reasonable default. "
        "Your output MUST be a valid Python list of one dictionary with these keys and corrected values, and nothing else."
    ),
    expected_output="A valid Python list of one dictionary with keys: 'dietary', 'cuisine', 'time', 'ingredients_to_use', 'ingredients_to_avoid'.",
    agent=input_parser
)

# Task 1: Brainstorm recipe ideas based on analyzed preferences
# This task is given to the 'recipe_brainstormer' agent.
# It uses the output from 'parse_and_correct_input_task' to generate recipe ideas. 
brainstorm_recipes_task = Task(
    description=(
        "Based on the inputs provided by the previous task, generate a few meal ideas. "
        "You must always produce a Python list of 3-5 dictionaries, each with keys: 'name', 'description', and 'tags', regardless of the quality or completeness of the input. "
        f"Do NOT generate any meal idea whose name matches any of these previous meal names: {read_meal_names()}. "
        "Interpret the input, infer missing details if necessary, and brainstorm 3-5 diverse and appealing dinner recipe ideas that are new and not previously suggested. "
        "For each idea, provide a catchy name, a brief description (1-2 sentences), and relevant tags (e.g., dietary, cuisine, time). "
        "If the input is empty, minimal, or malformed, use reasonable defaults or invent ideas, but always generate a valid output and never leave the output blank."
        "Do NOT include any commentary, explanation, or status update. "
    ),
    expected_output="A Python list of 3-5 dictionaries, each with keys: 'name', 'description', and 'tags'.",
    agent=recipe_brainstormer,
    context=[parse_and_correct_input_task]
)

# Task 2: Select and detail one meal suggestion
# This task is given to the 'meal_planner' agent.
# It uses the output from 'brainstorm_recipes_task' (recipe_ideas) to select one and provide details.
select_and_detail_meal_task = Task(
    description=(
        "From the brainstormed recipe ideas provided as a Python list of dictionaries from the previous task, interpret each dictionary which contains keys: 'name', 'description', and 'tags'. "
        "Select the best dinner idea from the list. Use the 'name' as the recipe name, 'description' as the dish summary, and 'tags' for dietary/cuisine/time context. "
        "For the selected recipe, return a formatted meal suggestion including: "
        "'Meal Suggestion' (recipe name), 'Ingredients' (bulleted list with approximate quantities for 4 servings), "
        "and 'Instructions' (clear, step-by-step numbered cooking directions). "
        "Always produce output, even if the input is empty, minimal, or unclear. Invent a reasonable recipe that fits the context if needed, but never leave the output blank."
    ),
    expected_output='A formatted meal suggestion including: "Meal Suggestion: [Recipe Name]", "Ingredients: [Bulleted list with quantities]", "Instructions: [Numbered steps]".',
    agent=meal_planner,
    context=[brainstorm_recipes_task]
)

# --- Create and Run the Crew ---

# Instantiate your crew with the defined agents and tasks.
# The process is sequential, meaning tasks run in the order they are defined.
dinner_crew = Crew(
    agents=[input_parser, recipe_brainstormer, meal_planner],
    tasks=[parse_and_correct_input_task, brainstorm_recipes_task, select_and_detail_meal_task],
    process=Process.sequential, # Tasks will execute one after another
    verbose=True, # Set to True to see detailed logs of agent execution and thought process
)

# --- Streamlit UI ---
st.title("AI Dinner Decider 🍽️")
st.write("Let me help you decide what to cook for dinner!")

with st.form("dinner_form"):
    dietary = st.text_input("Dietary preference (e.g., Vegetarian, Vegan, Gluten-Free, None):")
    cuisine = st.text_input("Cuisine preference (e.g., Italian, Mexican, Asian, Indian, None):")
    time = st.text_input("Desired cooking time (e.g., <30min, 30-60min, >60min):")
    ingredients_to_use = st.text_input("Ingredients to use (comma-separated, e.g., pasta, spinach):")
    ingredients_to_avoid = st.text_input("Ingredients to avoid (comma-separated, e.g., mushrooms):")
    submitted = st.form_submit_button("Get Dinner Suggestion")

def is_valid_output(output):
    return output and output.strip() != "[]" and output.strip() != ""

if submitted:
    with st.spinner("Planning your dinner..."):
        # Run the crew with initial inputs
        result = dinner_crew.kickoff(inputs={
            'dietary': dietary,
            'cuisine': cuisine,
            'time': time,
            'ingredients_to_use': ingredients_to_use,
            'ingredients_to_avoid': ingredients_to_avoid
        })

        # Retry logic for each task output (max 3 attempts)
        max_attempts = 3

        # Parsed & Corrected Input
        parsed_output = str(parse_and_correct_input_task.output)
        attempts = 1
        while not is_valid_output(parsed_output) and attempts < max_attempts:
            result = dinner_crew.kickoff(inputs={
                'dietary': dietary,
                'cuisine': cuisine,
                'time': time,
                'ingredients_to_use': ingredients_to_use,
                'ingredients_to_avoid': ingredients_to_avoid
            })
            parsed_output = str(parse_and_correct_input_task.output)
            attempts += 1

        # Brainstormed Recipes
        brainstormed_output = str(brainstorm_recipes_task.output)
        attempts = 1
        while not is_valid_output(brainstormed_output) and attempts < max_attempts:
            result = dinner_crew.kickoff(inputs={
                'dietary': dietary,
                'cuisine': cuisine,
                'time': time,
                'ingredients_to_use': ingredients_to_use,
                'ingredients_to_avoid': ingredients_to_avoid
            })
            brainstormed_output = str(brainstorm_recipes_task.output)
            attempts += 1

        # Selected Meal Suggestion
        meal_suggestion_output = str(select_and_detail_meal_task.output)
        attempts = 1
        while not is_valid_output(meal_suggestion_output) and attempts < max_attempts:
            result = dinner_crew.kickoff(inputs={
                'dietary': dietary,
                'cuisine': cuisine,
                'time': time,
                'ingredients_to_use': ingredients_to_use,
                'ingredients_to_avoid': ingredients_to_avoid
            })
            meal_suggestion_output = str(select_and_detail_meal_task.output)
            attempts += 1

    st.subheader("Parsed & Corrected Input")
    st.code(parsed_output, language='python')

    if not is_valid_output(parsed_output):
        st.warning("No valid parsed input was generated. Please check your input and try again.")
    else:
        st.subheader("Brainstormed Recipes")
        st.code(brainstormed_output, language='python')

        st.subheader("Selected Meal Suggestion")
        st.code(meal_suggestion_output, language='python')

        st.markdown("## Here is your dinner suggestion for today!")
        st.markdown(meal_suggestion_output)

        # --- Store the generated final meal name in a file, keeping only last 15 ---
        import re
        meal_name = None
        match = re.search(r"Meal Suggestion\s*:\s*(.+)", meal_suggestion_output)
        if match:
            meal_name = match.group(1).strip()
        else:
            meal_name = meal_suggestion_output.strip()

        # Read existing meal names, append new one, keep last 15
        file_path = "final_meal_name.txt"
        try:
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            lines = []
        lines.append(meal_name)
        lines = lines[-5:]
        with open(file_path, "w") as f:
            for line in lines:
                f.write(line + "\n")
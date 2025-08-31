# Main entry point for dinner recommender
# Warning control
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from litellm import completion

class LiteLLMOllama:
    def __init__(self, model, base_url):
        self.model = model
        self.base_url = base_url

    def __call__(self, prompt, **kwargs):
        return completion(
            model=self.model,
            api_base=self.base_url,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )["choices"][0]["message"]["content"]

load_dotenv() 
# Initialize Ollama LLM (make sure Ollama server is running)
llm = LiteLLMOllama(model="ollama/llama3", base_url="http://localhost:11434")


#

# Agent 1: Preference Analyst
# Role: Specializes in interpreting user requirements and dietary needs.
# Goal: To accurately extract and summarize all meal preferences and constraints.
# Backstory: An expert in food science and nutrition, ensuring all dietary and taste preferences are met.
preference_analyst = Agent(
    role='Dietary and Preference Analyst',
    goal='Always produce a summarized output of user dietary needs, cuisine preferences, cooking time, and ingredient availability for meal planning, even if user input is minimal, missing, or unclear.',
    backstory='You are an expert in food science and nutrition, skilled at inferring and summarizing dietary and taste preferences from even sparse or vague user input. You always return a valid output, filling in reasonable defaults when necessary, and never leave the output blank.',
    verbose=True,
    allow_delegation=False, # This agent does not delegate its core task of analysis
    llm=llm # Assign the configured LLM
)
# Agent 2: Recipe Brainstormer
# Role: Generates creative and diverse recipe ideas.
# Goal: To propose several meal options adhering to the analyzed preferences.
# Backstory: A seasoned chef with a global culinary background, known for innovative and delicious recipe concepts.


recipe_brainstormer = Agent(
    role='Creative Recipe Brainstormer',
    goal='Always generate 3-5 diverse and appealing meal ideas based on the analyzed preferences, interpreting the previous task\'s output even if it is minimal, incomplete, or not perfectly formatted.',
    backstory='You are a culinary expert with a vast knowledge of global cuisines and ingredients. You always return a valid list of recipe ideas, inferring reasonable defaults or inventing ideas if necessary, and never leave the output blank.',
    verbose=True,
    allow_delegation=False, # This agent can delegate if it needs further clarification or refinement
    llm=llm # Assign the configured LLM
)
# Agent 3: Meal Planner and Instruction Provider
# Role: Selects the best recipe and creates a simple, executable cooking plan.
# Goal: To deliver one clear, actionable dinner suggestion with ingredients and instructions.
# Backstory: A highly organized and practical meal prep specialist, making complex recipes simple for home cooks.
meal_planner = Agent(
    role='Daily Meal Planner and Instruction Provider',
    goal='Always select and detail a dinner meal suggestion, providing a concise cooking plan and ingredient list for a single dinner, even if recipe ideas or preferences are incomplete, minimal, or unclear.',
    backstory='You are an efficient and practical meal planner who always produces a valid meal suggestion, inferring reasonable defaults or inventing details when necessary. You never leave the output blank.',
    verbose=True,
    allow_delegation=False, # This agent delivers the final output and does not delegate its primary task
    llm=llm # Assign the configured LLM
)
#  Define Tasks  

# Task 1: Analyze user input for meal preferences
# This task is given to the 'preference_analyst' agent.
# It takes user_preferences as input and outputs a summarized list of requirements.
analyze_preferences_task = Task(
    description=(
        "Parse and understand the {user_dinner_preferences} which is a Python list of dictionaries and "
        "thoroughly analyze each dictionary, understand the user's intent, and use your expert knowledge about food and nutrition to infer and fix any entries that do not make sense, are ambiguous, or are missing. "
        "For example, correct spelling mistakes, clarify vague terms, and fill in reasonable defaults for missing information. "
        "If the input is empty, minimal, or unclear, you must still generate a valid Python list of dictionaries with reasonable default values for each key. "
        "After this expert review, generate and create a new Python list of dictionaries with keys: 'dietary', 'cuisine', 'time', 'ingredients_to_use', 'ingredients_to_avoid', reflecting your best understanding of the user's preferences and constraints. "
        "Always produce a valid output, never leave the output blank. Example: [{'dietary': 'Vegetarian', 'cuisine': 'Italian', 'time': '<30min', 'ingredients_to_use': ['pasta', 'spinach'], 'ingredients_to_avoid': ['mushrooms']}]"
    ),
    expected_output="A Python list of dictionaries, each with keys: 'dietary', 'cuisine', 'time', 'ingredients_to_use', 'ingredients_to_avoid', with spelling, logic, and defaults fixed by an expert. Always generate a valid output.",
    agent=preference_analyst
)

# Task 2: Brainstorm recipe ideas based on analyzed preferences
# This task is given to the 'recipe_brainstormer' agent.
# It uses the output from 'analyze_preferences_task' (preferences_summary) to generate recipe ideas. 
brainstorm_recipes_task = Task(
    description=(
        "You must always produce a Python list of 3-5 dictionaries, each with keys: 'name', 'description', and 'tags', regardless of the quality or completeness of the input. "
        "The input is a Python list of dictionaries, each with keys: 'dietary', 'cuisine', 'time', 'ingredients_to_use', 'ingredients_to_avoid'. Example input: "
        "[{'dietary': 'Vegetarian', 'cuisine': 'Mexican', 'time': '<30min', 'ingredients_to_use': ['beans', 'sweet potatoes'], 'ingredients_to_avoid': ['meat']}] "
        "Interpret the input, infer missing details if necessary, and brainstorm 3-5 diverse and appealing dinner recipe ideas. "
        "For each idea, provide a catchy name, a brief description (1-2 sentences), and relevant tags (e.g., dietary, cuisine, time). "
        "If the input is empty, minimal, or malformed, use reasonable defaults or invent ideas, but always generate a valid output and never leave the output blank. "
        "Example output: [{'name': 'Speedy Spinach & Tomato Pasta', 'description': 'A quick and flavorful pasta dish with fresh spinach and cherry tomatoes, seasoned with garlic and herbs.', 'tags': ['Vegetarian', 'Italian', '<30min']}, ...]"
    ),
    expected_output="A Python list of 3-5 dictionaries, each with keys: 'name', 'description', and 'tags'. Always generate a valid output.",
    agent=recipe_brainstormer,
    context=[analyze_preferences_task]
)

# Task 3: Select and detail one meal suggestion
# This task is given to the 'meal_planner' agent.
# It uses the output from 'brainstorm_recipes_task' (recipe_ideas) to select one and provide details.
select_and_detail_meal_task = Task(
    description=(
        "From the brainstormed recipe ideas provided as a Python list of dictionaries from the previous task, interpret each dictionary which contains keys: 'name', 'description', and 'tags'. "
        "For example, the input may look like: "
        "[{'name': 'Vegan Black Bean and Sweet Potato Chili', 'description': 'A hearty, plant-based chili made with black beans, roasted sweet potatoes, and smoky spices.', 'tags': ['Vegetarian', 'Mexican', '<30min']}, "
        "{'name': 'Grilled Salmon with Lemon-Herb Quinoa and Steamed Asparagus', 'description': 'A bright and flavorful dish featuring grilled salmon, creamy quinoa infused with lemon zest and herbs, and tender asparagus.', 'tags': ['Seafood', 'Mediterranean', '10-30min']}, ...] "
        "Select the best dinner idea from the list. Use the 'name' as the recipe name, 'description' as the dish summary, and 'tags' for dietary/cuisine/time context. "
        "For the selected recipe, return a formatted meal suggestion including: "
        "'Meal Suggestion' (recipe name), 'Ingredients' (bulleted list with approximate quantities for 4 servings), "
        "and 'Instructions' (clear, step-by-step numbered cooking directions). "
        "Always produce output, even if the input is empty, minimal, or unclear. Invent a reasonable recipe that fits the context if needed, but never leave the output blank."
    ),
    expected_output='A formatted meal suggestion including: "Meal Suggestion: [Recipe Name]", "Ingredients: [Bulleted list with quantities]", "Instructions: [Numbered steps]". Always generate a valid output.',
    agent=meal_planner,
    context=[brainstorm_recipes_task]
)

# --- Create and Run the Crew ---

# Instantiate your crew with the defined agents and tasks.
# The process is sequential, meaning tasks run in the order they are defined.
dinner_crew = Crew(
    agents=[preference_analyst, recipe_brainstormer, meal_planner],
    tasks=[analyze_preferences_task, brainstorm_recipes_task, select_and_detail_meal_task],
    process=Process.sequential, # Tasks will execute one after another
    verbose=True, # Set to True to see detailed logs of agent execution and thought process
)

# Get user input for dinner preferences
print("Hello! I'm your AI Dinner Decider. Let's plan your meal.")
print("Please answer the following questions about your dinner preferences:")

dietary = input("Dietary preference (e.g., Vegetarian, Vegan, Gluten-Free, None): ")
cuisine = input("Cuisine preference (e.g., Italian, Mexican, Asian, Indian, None): ")
time = input("Desired cooking time (e.g., <30min, 30-60min, >60min): ")
ingredients_to_use = input("Ingredients to use (comma-separated, e.g., pasta, spinach): ")
ingredients_to_avoid = input("Ingredients to avoid (comma-separated, e.g., mushrooms): ")

user_dinner_preferences = [{
    'dietary': dietary.strip(),
    'cuisine': cuisine.strip(),
    'time': time.strip(),
    'ingredients_to_use': [i.strip() for i in ingredients_to_use.split(',') if i.strip()],
    'ingredients_to_avoid': [i.strip() for i in ingredients_to_avoid.split(',') if i.strip()]
}]

# Kick off the crew with the initial input
print("\nStarting the dinner planning crew...\n")
result = dinner_crew.kickoff(inputs={'user_dinner_preferences': user_dinner_preferences})

# Print outputs of each task
print("\n--- Task Outputs ---")
print("\nAnalyzed Preferences:\n", analyze_preferences_task.output)
print("\nBrainstormed Recipes:\n", brainstorm_recipes_task.output)
print("\nSelected Meal Suggestion:\n", select_and_detail_meal_task.output)


# Print the final dinner suggestion
print("\n\n##################################")
print("## Here is your dinner suggestion for today!")
print("##################################\n")
print(result)
# from IPython.display import Markdown
# Markdown(result.tasks_output)
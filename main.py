from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.agent import ReActAgent
# from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.tools import ToolMetadata, QueryEngineTool

from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize LLMs
llm = Ollama(model="mistral", request_timeout=3600.0)
quote_llm = Ollama(model="llama2", request_timeout=3600.0)

# Document Retrieval for RAG
documents = SimpleDirectoryReader("./data").load_data()
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)


class CustomTool:
    def __init__(self, name, description, function):
        self.metadata = ToolMetadata(name=name, description=description)
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


# Define Tools
def calculate_macros_tool(input_data):
    """Tool to calculate macros deterministically."""
    weight = input_data["weight"]
    height = input_data["height"]
    age = input_data["age"]
    gender = input_data["gender"]
    goal = input_data["goal"]

    # Harris-Benedict Equation
    calories = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)

    if "gain" in goal.lower():
        protein = calories * 0.3 / 4
        carbs = calories * 0.4 / 4
        fats = calories * 0.3 / 9
    elif "loss" in goal.lower():
        protein = calories * 0.4 / 4
        carbs = calories * 0.3 / 4
        fats = calories * 0.3 / 9
    else:
        protein, carbs, fats = 0, 0, 0  # Placeholder for other goals

    return {
        "calories": int(calories),
        "protein": int(protein),
        "carbs": int(carbs),
        "fats": int(fats),
    }

macro_calculator_tool = CustomTool(
    name="macro_calculator",
    description="Calculates macros based on user profile and goals.",
    function=calculate_macros_tool,
)


def motivational_quote_tool():
    """Tool to generate a motivational quote."""
    prompt = "Generate an inspiring fitness-related motivational quote."
    response = quote_llm(prompt)
    print("Response from LLM", response)
    return response["response"]

motivational_quote_tool_instance = CustomTool(
    name="motivational_quote",
    description="Generates a motivational quote.",
    function=motivational_quote_tool,
)


# Register Tools
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="nutrition_documents",
            description="Provides context for nutrition and workout planning.",
        ),
    ),
    macro_calculator_tool,
    motivational_quote_tool_instance,
]

# Initialize ReAct Agent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# User Profile Management
def get_profile(profile_id):
    """Retrieve user profile (mocked for now)."""
    return {
        "id": profile_id,
        "general": {
            "name": "John Doe",
            "age": 30,
            "weight": 75,
            "height": 180,
            "gender": "Male",
            "activity_level": "Moderately Active",
        },
        "goals": ["Muscle Gain"],
        "notes": [],
        "nutrition": {},
    }

def update_personal_info(profile, section, **kwargs):
    """Update user profile details."""
    if section in profile:
        profile[section].update(kwargs)
    else:
        profile[section] = kwargs
    return profile

# Notes Management
def get_notes(profile_id):
    """Retrieve notes for a profile."""
    return [{"_id": 1, "text": "I can't do squats due to a knee injury."}]

def add_note(note_text, profile_id):
    """Add a new note."""
    notes = get_notes(profile_id)
    new_note = {"_id": len(notes) + 1, "text": note_text}
    notes.append(new_note)
    return notes

# AI Integration
def calculate_macros(profile):
    """Calculate macros deterministically using the agent."""
    tool_input = {
        "weight": profile["general"]["weight"],
        "height": profile["general"]["height"],
        "age": profile["general"]["age"],
        "gender": profile["general"]["gender"],
        "goal": ", ".join(profile["goals"]),
    }
    
    # Convert tool_input to a query string
    query_str = (
        f"Calculate macros for a {tool_input['age']} year old "
        f"{tool_input['gender']} with weight {tool_input['weight']} kg, "
        f"height {tool_input['height']} cm, and goal: {tool_input['goal']}."
    )
    
    # Pass query string to the agent
    response = agent.query(query_str)

    # Extract and format the response
    if hasattr(response, "response"):
        # Parse the agent's response into a dictionary
        text = response.response
        try:
            macros = {
                "protein": int(text.split("Protein - ")[1].split(" grams")[0]),
                "carbs": int(text.split("Carbohydrates - ")[1].split(" grams")[0]),
                "fat": int(text.split("Fat - ")[1].split(" grams")[0]),
                "fiber": int(text.split("Fiber - ")[1].split(" grams")[0]),
            }
            return macros
        except Exception:
            raise ValueError("Failed to parse the agent's response into macros.")
    elif isinstance(response, dict):
        return response
    else:
        raise ValueError("Unexpected response format received from the agent.")


def ask_ai(profile, question):
    """Handles user custom queries via RAG and adds a motivational quote."""
    # Step 1: Use RAG to generate the primary response
    prompt = f"""
    User Profile:
    Name: {profile['general']['name']}
    Age: {profile['general']['age']}
    Gender: {profile['general']['gender']}
    Weight: {profile['general']['weight']}
    Height: {profile['general']['height']}
    Activity Level: {profile['general']['activity_level']}
    Goals: {', '.join(profile['goals'])}
    Notes: {', '.join([note['text'] for note in get_notes(profile['id'])])}

    Question: {question}
    """
    rag_response = agent.query(prompt)

    # # Step 2: Add a motivational quote
    # prompt2 = "Generate an inspiring fitness-related motivational quote."
    # motivational_quote = agent.query(prompt2)
    
    # Combine responses
    final_response = {
        "rag_response": rag_response["response"],
        # "motivational_quote": motivational_quote,
    }
    return final_response

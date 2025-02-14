import julia
import julia.Main
import os
import functools

# Singleton pattern for Julia runtime initialization
class JuliaRuntime:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = julia.Julia()
        return cls._instance

# Initialize Julia runtime lazily
jl = JuliaRuntime.get_instance()
julia_script_path = os.getenv("JULIA_SCRIPT_PATH", "../julia_core/memory_model.jl")

# Check if the Julia script exists before including it
if not os.path.exists(julia_script_path):
    raise FileNotFoundError(f"Julia script not found: {julia_script_path}")

jl.include(julia_script_path)

# Simple in-memory cache to avoid redundant storage operations
memory_cache = {}

def retrieve_memory(query_vector, top_k=5):
    """
    Calls the Julia memory retrieval function.
    
    Args:
        query_vector (list): The input query vector.
        top_k (int): Number of nearest neighbors to retrieve.
    
    Returns:
        list: Retrieved memory items.
    """
    if not isinstance(query_vector, list) or not all(isinstance(i, (int, float)) for i in query_vector):
        raise ValueError("query_vector must be a list of numerical values")
    
    try:
        return jl.memory_retrieval(query_vector, top_k)
    except Exception as e:
        raise RuntimeError(f"Error in Julia retrieval: {str(e)}")

def store_memory(key, value):
    """
    Calls the Julia memory storage function with caching to avoid redundant storage.
    
    Args:
        key (list): The key vector.
        value (list): The value vector.
    
    Returns:
        str: Confirmation message.
    """
    key_tuple = tuple(key)  # Convert list to tuple for hashability
    if key_tuple in memory_cache and memory_cache[key_tuple] == value:
        return "Memory already stored, skipping redundant operation"
    
    try:
        jl.store_memory(key, value)
        memory_cache[key_tuple] = value  # Update cache
        return "Memory stored successfully"
    except Exception as e:
        raise RuntimeError(f"Error in Julia storage: {str(e)}")

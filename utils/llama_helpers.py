import ollama

def run_llama(prompt):
    """
    Use Ollama to run a prompt on your Llama model.=
    """
    try:
        response = ollama.generate(model="llama3", prompt=prompt)
        # response is typically a dict with {'text': '...'}
        return response.get('text', str(response))
    except Exception as e:
        print(f"Error: {e}")
        return ""


def extract_structure(structured_response):
    """
    Assuming Llama responded with something like:
      'Metric: 2 meters, Semantic: vase, Predicate: left of'
    parse out metric, semantic, and predicate.
    """
    metric = None
    semantic = None
    predicate = None

    if "Metric:" in structured_response:
        metric_start = structured_response.find("Metric:") + len("Metric:")
        metric_end = structured_response.find(",", metric_start)
        if metric_end == -1:
            metric_end = len(structured_response)
        metric = structured_response[metric_start:metric_end].strip()

    if "Semantic:" in structured_response:
        semantic_start = structured_response.find("Semantic:") + len("Semantic:")
        semantic_end = structured_response.find(",", semantic_start)
        if semantic_end == -1:
            semantic_end = len(structured_response)
        semantic = structured_response[semantic_start:semantic_end].strip()

    if "Predicate:" in structured_response:
        predicate_start = structured_response.find("Predicate:") + len("Predicate:")
        predicate_end = structured_response.find("'", predicate_start)
        if predicate_end == -1:
            predicate_end = len(structured_response)
        predicate = structured_response[predicate_start:predicate_end].strip()

    return metric, semantic, predicate
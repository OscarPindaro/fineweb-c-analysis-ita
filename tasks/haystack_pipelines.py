from src.haystack.topic_pipeline import PromptVariables, TopicExtractionPipeline
from pathlib import Path
from haystack import Pipeline

def create_topic_extraction_pipeline(product, model_id, system_prompt_path, ask_prompt_path, assistant_start="<classe=\"", stream_reaponse=False):
    system_prompt_path= Path(system_prompt_path)
    ask_prompt_path = Path(ask_prompt_path)
    assert system_prompt_path.exists() and system_prompt_path.is_file()
    assert ask_prompt_path.exists() and ask_prompt_path.is_file()
    print(ask_prompt_path)
    system_prompt = system_prompt_path.read_text()
    ask_prompt= ask_prompt_path.read_text()
    
    
    pipeline: Pipeline = TopicExtractionPipeline.create_pipeline(
        model_id=model_id,
        system_prompt=system_prompt,
        ask_prompt=ask_prompt,
        assistant_start=assistant_start,
        stream_response=stream_reaponse
    )
    product = Path(product)
    product.parent.mkdir(exist_ok=True, parents=True)
    pipeline.dump(product.open("w"))
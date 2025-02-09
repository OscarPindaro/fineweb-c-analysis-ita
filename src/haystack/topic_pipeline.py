from pathlib import Path
from haystack import Document
from dataclasses import dataclass, field, asdict
from typing import List
import json
from typing import TypeVar, Type

from haystack import Pipeline
from haystack.dataclasses import ChatMessage, ChatRole
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.converters import OutputAdapter
from tqdm import tqdm

T = TypeVar('T', bound='PromptVariables')

@dataclass
class PromptVariables:
    examples: List[Document] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """
        Serialize the PromptVariables instance to a JSON string.
        
        Returns:
            str: JSON string representation of the instance
        """
        return json.dumps(asdict(self), default=lambda o: o.__dict__, indent=2)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create a new PromptVariables instance from a JSON string.
        
        Args:
            json_str: JSON string representation of PromptVariables
            
        Returns:
            PromptVariables: New instance created from JSON data
        """
        data = json.loads(json_str)
        # Reconstruct Document objects if necessary
        if 'examples' in data:
            data['examples'] = [Document(**doc) if isinstance(doc, dict) else doc 
                              for doc in data['examples']]
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save PromptVariables to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, default=lambda o: o.__dict__)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PromptVariables':
        """Load PromptVariables from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'examples' in data:
                data['examples'] = [Document(**doc) if isinstance(doc, dict) else doc 
                                  for doc in data['examples']]
            return cls(**data)
    
from typing import List
from haystack import component
from haystack import Pipeline
from haystack.dataclasses import GeneratedAnswer

@component
class TopicExtractionPipeline:
    
    def __init__(self, pipeline_path: Path | str, update_categories: bool = True, ret_filled_prompt: bool = False):
        self.pipeline_path: Path | str = pipeline_path
        if self.pipeline_path is not None:
            self.pipeline_path = Path(self.pipeline_path)
            assert self.pipeline_path.exists()
            assert self.pipeline_path.is_file()
        self.pipeline: Pipeline = Pipeline.load(self.pipeline_path.open("r"))
        self.update_categories: bool = update_categories
        self.ret_filled_prompt: bool = ret_filled_prompt
    
    @component.output_types(categories=List[str], answers=List[List[GeneratedAnswer]], prompts=List[List[ChatMessage]] | None)
    def run(self, input_samples: List[Document], prompt_vars: PromptVariables):
        categories: List[str] = []
        answers: List[List[GeneratedAnswer]] = []
        builder_outs: List[List[ChatMessage]] = []
        for input_sample in tqdm(input_samples):
            out = self.pipeline.run(data={
                "InputDoc":{"value":input_sample},
                "Builder":{
                    "categories":prompt_vars.categories,
                    "examples":prompt_vars.examples}
                }, include_outputs_from={"Builder"}
            )
            category=out["AnswerParser"]["answers"][0].data
            if self.update_categories is True and category not in prompt_vars.categories:
                prompt_vars.categories.append(prompt_vars)
            categories.append(category)
            answers.append(out["AnswerParser"]["answers"])
            builder_outs.append(out["Builder"]["prompt"])
            
        to_ret = {
            "categories":categories,
            "answers":answers
        }
        if self.ret_filled_prompt:
            to_ret["prompts"] = builder_outs
        return to_ret
    
    @classmethod
    def create_pipeline(cls, model_id, system_prompt, ask_prompt, assistant_start="<classe=\"", stream_response: bool = False) -> Pipeline:
        """I built my pipeline with gemma, llama and deepseek in my mind. Other llms are supported, will use the llama structure (so with system prompt divided by user prompt.
        My code also assumes that parsing is done with my weird xlm sintax. If that's not the case, just change the answer pattern and the stop token.
        """
        
        gemma_messages = [ChatMessage.from_user(system_prompt+ask_prompt), ChatMessage.from_assistant(assistant_start)]
        llama_messages = [ChatMessage.from_system(system_prompt), ChatMessage.from_user(ask_prompt), ChatMessage.from_assistant(assistant_start)]
        deepseek_messages = [ChatMessage.from_system(system_prompt), ChatMessage.from_user(ask_prompt)]
        
        stop_tokens = ["/>"]
        answer_pattern = "(.*)\""
        if "gemma" in model_id:
            messages = gemma_messages
        elif "deepseek" in model_id:
            messages = deepseek_messages
            stop_tokens = None
            # deepseek needs to think, so i need to capture directly
            answer_pattern="<classe=\"(.*)\""
        elif "llama" in model_id:
            messages = llama_messages
        else: 
            messages = llama_messages
        
        # let's create the haystack components
        required_variables = ["campione", "categories"]
        variables = ["campione", "examples", "categories"]

        prompt_builder = ChatPromptBuilder(messages,required_variables=required_variables, variables=variables)
        generator = OllamaChatGenerator(
            model=model_id,
            generation_kwargs={
                "stop":stop_tokens,
                "seed":42,
                "num_ctx":8192+4096
                },
            streaming_callback= (lambda chunk: print(chunk.content, end="", flush=True) ) if stream_response is True else None
        )

        answer_parser = AnswerBuilder(
            pattern=answer_pattern
        )
        # answer_parser = AnswerBuilder(
        #     pattern="(.*)"
        # )
        input_doc_joiner= BranchJoiner(Document)
        doc_to_str = OutputAdapter("{{doc.content}}", str)
        
        # add and connect components
        pipeline = Pipeline()
        pipeline.add_component("InputDoc", input_doc_joiner)
        pipeline.add_component("DocToStr", doc_to_str)
        pipeline.add_component("Builder", prompt_builder)
        pipeline.add_component("Generator", generator)
        pipeline.add_component("AnswerParser", answer_parser)

        pipeline.connect("InputDoc.value", "Builder.campione")
        pipeline.connect("InputDoc.value", "DocToStr")

        pipeline.connect("Builder.prompt", "Generator.messages")

        pipeline.connect("DocToStr", "AnswerParser.query")
        pipeline.connect("Generator", "AnswerParser.replies")
        return pipeline
    
    
DEFAULT_SYSTEM_PROMPT="""Sei un assistente specializzato nell'analisi e classificazione di testi in italiano. Il tuo compito è duplice:

1. Comprendere il livello qualitativo del contenuto informativo del testo, basandoti sulla seguente scala:
   - Problematic Content: contenuti inappropriati (pornografia, gambling, testo mal formattato)
   - None: assenza di contenuto informativo (es. pubblicità, post social)
   - Minimal: contenuto con minima valenza informativa non intenzionale
   - Basic: contenuto con discreto valore informativo
   - Good: contenuto ben strutturato con chiaro intento educativo
   - Basic: contenuto con elevato valore informativo e ottima strutturazione

2. Identificare una categoria tematica di alto livello che rappresenti l'argomento principale del testo.

CONTESTO OPERATIVO:
- Hai accesso a un elenco di parole chiave di basso livello estratte dal testo
- Hai accesso a un elenco di categorie tematiche già utilizzate in precedenza
- Puoi sia utilizzare categorie esistenti che crearne di nuove quando necessario

REGOLE DI CATEGORIZZAZIONE:
- Usa categorie ampie e generali (es. "Medicina", "Sport", "Tecnologia")
- Mantieni consistenza con le categorizzazioni precedenti
- Crea nuove categorie solo quando strettamente necessario
- Usa sempre singolare per le categorie (es. "Calcio" non "Calcistica")
- Usa nomi semplici e diretti (es. "Politica" non "Scienze Politiche")

OUTPUT:
Devi sempre rispondere utilizzando esclusivamente questo formato XML:
<classe="CATEGORIA" />

Dove CATEGORIA è la categoria tematica identificata.

ESEMPI DI CATEGORIZZAZIONE:
- Testi su malattie, cure, farmaci → "Medicina"
- Testi su partite, campionati → "Calcio"
- Testi su prodotti in vendita → "Pubblicità"
- Testi su smartphone, computer → "Tecnologia"
- Testi su ricette, cucina → "Gastronomia"
- Testi pubblicitari in cui singole persone promuovono il proprio lavoro-> "Autopromozione"

{% if examples%}
## Esempi
{% for ex in examples%}
### Esempio {{loop.index}}
Testo: {{ex.content}}
{% if ex.meta['quality']%}Qualità: {{ex.meta['quality']}}
{%endif%}{% if ex.meta['keywords']%}Parole Chiave: {{ex.meta['keywords']}}
{%endif%}Categoria: <classe="{{ex.meta['category']}}" />

---
{% endfor %}
{%endif%}
Categorie Esistenti:
{% for cat in categories%}
- "{{cat}}"
{% endfor %}
NOTA IMPORTANTE:
Prima di creare una nuova categoria, verifica sempre se è possibile utilizzare una categoria esistente nell'elenco fornito. La creazione di nuove categorie deve essere l'ultima risorsa quando nessuna categoria esistente è appropriata.
"""

DEFAULT_ASK_PROMPT = """Campione:
---
Testo: {{campione.content}}
{% if campione.meta['quality']%}Qualità: {{campione.meta['quality']}}
{%endif%}{% if campione.meta['keywords']%}Parole Chiave: {{campione.meta['keywords']}}
{%endif%}
"""
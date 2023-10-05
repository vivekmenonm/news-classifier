from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])

# Load pipeline
model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

def keywords(text):
    keyphrases = extractor(text)
    return keyphrases


# input_text = "Former Australia cricketer Tom Moody has said that the biggest difference between India and Australia in the ongoing series has been pacer Josh Hazlewood and spinner Adam Zampa. In the two ODIs, which Australia have won, Hazlewood has taken five wickets, while Zampa has picked six wickets. Zampa is the unsung hero of this whole thing,Moody further stated. "
# a = keywords(input_text)
# print(a)
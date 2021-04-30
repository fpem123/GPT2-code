from pydantic import BaseModel, Field
from transformers import ReformerModelWithLMHead, ReformerTokenizer

tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
model = ReformerModelWithLMHead.from_pretrained('google/reformer-crime-and-punishment')


##
# GPT-2 generator.
# Make java code!.
def mk_crime_punish(text, length, how_many, top_p, top_k, do_sample):
    try:
        input_ids = tokenizer.encode(text, return_tensors='pt')

        min_length = len(input_ids.tolist()[0])
        length += min_length

        length = length if length > 0 else 1
        top_k = top_k if top_k > 0 else 10
        top_p = top_p if top_p > 0 else 0.5

        # model generating
        sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=do_sample,
                                        max_length=length,
                                        top_p=top_p,
                                        top_k=top_k,
                                        num_return_sequences=how_many)

        result = dict()

        for idx, sample_output in enumerate(sample_outputs):
            story = tokenizer.decode(sample_output, skip_special_tokens=True)
            print(story)
            result[idx] = story

        return result

    except Exception as e:
        print('Error occur in python code generating!', e)
        return {'error': e}, 500


class Input(BaseModel):
    text: str = Field(
        ...,
        title="Text Input",
        description="The input text to use as basis to generate text.",
        max_length=500,
    )

    length: int = Field(
        30,
        ge=1,
        le=200,
        description="The maximum length of the sequence to be generated.",
    )

    how_many: int = Field(
        3,
        ge=1,
        le=5,
        description="How many generation?",
    )

    top_k: int = Field(
        50,
        ge=1,
        le=100,
        description="top_k",
    )

    top_p: float = Field(
        0.8,
        ge=0.1,
        le=1.0,
        description="top_p",
    )

    do_sample: bool = Field(
        True,
        description="Whether or not to use sampling ; use greedy decoding otherwise.",
    )


class Output(BaseModel):
    message: dict = Field(...)


def generate_crime_punish(input: Input) -> Output:
    res = mk_crime_punish(
        input.text,
        input.length,
        input.how_many,
        input.top_p,
        input.top_k,
        input.do_sample
    )

    return Output(message=res)

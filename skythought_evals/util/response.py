from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
from typing import List, Dict

def regroup_list_of_dicts(data: List[Dict]) -> Dict[str, List]:
    regrouped = defaultdict(list)
    for entry in data:
        for key, value in entry.items():
            regrouped[key].append(value)
    return dict(regrouped)

@dataclass
class Response:
    response: List[str]
    num_completion_tokens: List[int]
    num_input_tokens: int
    index: Optional[int] = None
    num_time_spend: List[float] = field(default_factory=list)
    num_correct_tokens: List = field(default_factory=list)
    num_try_correct: List[int] = field(default_factory=list)

    @classmethod
    def from_spe_response(cls, response, idx) -> "Response":
        new_r = regroup_list_of_dicts(response) 
        return cls(
            index = idx,
            response=new_r['generated_text'],
            num_completion_tokens=new_r['num_tokens'],
            num_input_tokens=new_r["question"][0],
            num_time_spend=new_r['generation_time'],
            num_correct_tokens=new_r['correct_tokens'],
            num_try_correct=new_r['try_correct_num']
        )

    @classmethod
    def from_ray_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a rayllm response.

        Args:
            response: Ray response object containing generated text and token information

        Returns:
            Responses: New instance initialized with Ray response data
        """

        if isinstance(response["generated_text"], list):
            # n > 1 samples
            response_texts = response["generated_text"]
            num_completion_tokens = [
                int(response["num_generated_tokens"][i])
                for i in range(len(response["num_generated_tokens"]))
            ]
        else:
            response_texts = [response["generated_text"]]
            num_completion_tokens = [int(response["num_generated_tokens"])]
        return cls(
            response=response_texts,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=int(response["num_input_tokens"]),
            index=response["index"],
        )

    @classmethod
    def from_openai_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from an OpenAI response.

        Args:
            response: OpenAI response object containing message content and token information

        Returns:
            Responses: New instance initialized with OpenAI response data
        """
        return cls(
            response=[
                response.choices[i].message.content
                for i in range(len(response.choices))
            ],
            num_completion_tokens=[
                response.usage.completion_tokens if i == 0 else 0
                for i in range(len(response.choices))
            ],
            num_input_tokens=response.usage.prompt_tokens,
        )

    @classmethod
    def from_vllm_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a vLLM response.

        Args:
            response: vLLM response object containing output text and token information

        Returns:
            Responses: New instance initialized with vLLM response data
        """
        response_texts = [
            response.outputs[i].text for i in range(len(response.outputs))
        ]
        num_completion_tokens = [
            len(response.outputs[i].token_ids) for i in range(len(response.outputs))
        ]
        return cls(
            response=response_texts,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=len(response.prompt_token_ids),
        )

    @classmethod
    def from_spe_decoding_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a vLLM response.

        Args:
            response: vLLM response object containing output text and token information

        Returns:
            Responses: New instance initialized with vLLM response data
        """
        response_texts = [
            response.outputs[i].text for i in range(len(response.outputs))
        ]
        num_completion_tokens = [
            len(response.outputs[i].token_ids) for i in range(len(response.outputs))
        ]
        correct_tokens = response.metrics.spec_token_acceptance_counts
        return cls(
            num_correct_tokens=correct_tokens,
            response=response_texts,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=len(response.prompt_token_ids),
        )


@dataclass
class SingleParsedResponse:
    content: str
    correctness: Optional[bool] = None
    reason: Optional[str] = None

    def to_dict(self):
        return {
            "content": self.content,
            "correctness": self.correctness,
            "reason": self.reason,
        }

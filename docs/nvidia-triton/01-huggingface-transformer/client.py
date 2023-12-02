from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from transformers import AutoTokenizer
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client() -> InferenceServerClient:
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_embedder_ensemble(text: str) -> np.ndarray:
    triton_client = get_client()
    text = np.asarray([text.encode("utf-8")], dtype=object)

    input_text = InferInput("TEXTS", shape=list(text.shape), datatype=np_to_triton_dtype(text.dtype))
    input_text.set_data_from_numpy(text, binary_data=True)

    infer_output = InferRequestedOutput("EMBEDDINGS", binary_data=True)

    query_response = triton_client.infer(
        model_name="ensemble-onnx",
        inputs=[input_text],
        outputs=[infer_output],
    )
    embeddings = query_response.as_numpy("EMBEDDINGS")[0]

    return embeddings


def call_triton_tokenizer(text: str) -> Tuple[np.ndarray, np.ndarray]:
    triton_client = get_client()
    text = np.asarray([text.encode("utf-8")], dtype=object)

    input_text = InferInput("TEXTS", shape=list(text.shape), datatype=np_to_triton_dtype(text.dtype))
    input_text.set_data_from_numpy(text, binary_data=True)

    query_response = triton_client.infer(
        model_name="python-tokenizer",
        inputs=[input_text],
        outputs=[
            InferRequestedOutput("INPUT_IDS", binary_data=True),
            InferRequestedOutput("ATTENTION_MASK", binary_data=True),
        ],
    )
    input_ids = query_response.as_numpy("INPUT_IDS")[0]
    attention_mask = query_response.as_numpy("ATTENTION_MASK")[0]

    return input_ids, attention_mask


def main():
    texts = ["I am Groot", "You are Groot", "We are Groot", "What the hell are you doing here?"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(texts[0], padding="max_length", max_length=16, truncation=True)

    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    _input_ids, _attention_mask = call_triton_tokenizer(texts[0])

    assert np.array_equal(input_ids, _input_ids)
    assert np.array_equal(attention_mask, _attention_mask)

    embeddings = torch.tensor([call_triton_embedder_ensemble(text).tolist() for text in texts])

    distances = torch.cdist(x1=embeddings, x2=embeddings, p=2)
    print(distances)


if __name__ == "__main__":
    main()

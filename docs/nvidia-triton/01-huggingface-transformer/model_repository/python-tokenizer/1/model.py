import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/assets/bert-base-uncased-tokenizer",
            local_files_only=True,
        )

    def tokenize(self, texts):
        encoded = self.tokenizer(texts, padding="max_length", max_length=16, truncation=True, return_tensors=None)
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)

        return input_ids, attention_mask

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [text.decode() for text in texts]

            input_ids, attention_mask = self.tokenize(texts)

            output_tensor_tokens = pb_utils.Tensor("INPUT_IDS", input_ids)
            output_tensor_masks = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor_tokens, output_tensor_masks]))
        return responses

"""Still taken to large extent from https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682
but works with tokens"""

from transformers import T5Tokenizer
import numpy as np

noise_density = 0.15
mean_noise_span_length = 3


def random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
    """Partition a sequence of items randomly into non-empty segments.
    Args:
        num_items: an integer scalar > 0
        num_segments: an integer scalar in [1, num_items]
    Returns:
        a Tensor with shape [num_segments] containing positive integers that add
        up to num_items
    """
    mask_indices = np.arange(num_items - 1) < (num_segments - 1)
    np.random.shuffle(mask_indices)
    first_in_segment = np.pad(mask_indices, [[1, 0]])
    segment_id = np.cumsum(first_in_segment)
    # count length of sub segments assuming that list is sorted
    _, segment_length = np.unique(segment_id, return_counts=True)
    return segment_length


def apply_mask(tokenized_text: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tokenized_text = np.array(tokenized_text)
    len_input = len(tokenized_text)
    tokens_to_mask = round(len_input * noise_density)
    num_spans = round(tokens_to_mask / mean_noise_span_length)

    mask_lengths = random_segmentation(tokens_to_mask, num_spans)
    offset = np.random.choice([0, 1])  # decides if sentence ends with masked part or not
    non_mask_lengths = random_segmentation(len_input - tokens_to_mask, num_spans + offset)
    interleaved_span_lengths = np.empty(mask_lengths.size + non_mask_lengths.size, dtype=mask_lengths.dtype)
    interleaved_span_lengths[0::2] = non_mask_lengths
    interleaved_span_lengths[1::2] = mask_lengths

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((len_input,), dtype=np.int8)

    span_start_indicator[span_starts] = True

    span_num = np.cumsum(span_start_indicator)
    input_noise_mask = np.equal(span_num % 2, 1)
    label_noise_mask = ~input_noise_mask

    input_noise_mask = input_noise_mask.astype(np.int8)
    label_noise_mask = label_noise_mask.astype(np.int8)

    input_sentinel_ids = create_sentinel_ids(input_noise_mask)
    label_sentinel_ids = create_sentinel_ids(label_noise_mask)

    masked_input_text = np.where(input_sentinel_ids != 0, input_sentinel_ids, np.array(tokenized_text))
    masked_input_text = masked_input_text[masked_input_text >= 0]

    masked_label_text = np.where(label_sentinel_ids != 0, label_sentinel_ids, np.array(tokenized_text))
    masked_label_text = masked_label_text[masked_label_text >= 0]
    return masked_input_text, masked_label_text


def create_sentinel_ids(noise_mask: np.ndarray) -> np.ndarray:
    start_indices = noise_mask - np.roll(noise_mask, 1, axis=-1) * noise_mask
    sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
    sentinel_ids = np.where(sentinel_ids != 0, (32100 - sentinel_ids), 0)
    sentinel_ids -= noise_mask - start_indices
    return sentinel_ids


def main():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    text = "Theoden, son of Thengel and Morwen Steelsheen, was the seventeenth King of Rohan, last of the Second Line of the royal House of Eorl."
    tokenized_text = tokenizer(text)
    masked_input, masked_output = apply_mask(tokenized_text.input_ids)
    return


main()


from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from sklearn.model_selection import train_test_split


model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id



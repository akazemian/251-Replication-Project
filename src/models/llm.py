from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

class LLM:
    def __init__(self, model_name:str, num_layers:int, max_window_size:int):
        self.model_name = model_name
        self.num_layers = num_layers 
        self.max_window_size = max_window_size

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def get_token_ids(self, input_string):
        """Tokenize the input string and return token IDs."""
        return self.tokenizer(input_string, return_tensors='pt')['input_ids'][0]
            
    def get_embeddings(self, identifier: str, layer: int, token_ids: torch.Tensor, batch_size: int):
        device = self.model.device
        token_ids = token_ids.to(device)

        # ────────────────────────────────────────────────────────────────
        # 0) Make sure our window size never exceeds the model's max context
        if self.max_window_size > len(token_ids):
            warnings.warn(
                'context length is larger than the number of tokens, setting context_length = number of tokens'
            )
            self.max_window_size = len(token_ids)
        # ────────────────────────────────────────────────────────────────

        print('----------max window size:----------',self.max_window_size)
        token_embeddings = []
        for i in tqdm(range(0, len(token_ids), batch_size)):
            batch_end = min(i + batch_size, len(token_ids))
            windows, attention_masks, position_ids_list = [], [], []

            for j in range(i, batch_end):
                # 1) slice so len(window) ≤ max_window_size
                start = max(0, j - self.max_window_size + 1)
                window = token_ids[start : j + 1]   # length = min(j+1, max_window_size)

                # 2) non-negative padding
                pad_len = max(self.max_window_size - window.size(0), 0)
                pad     = lambda n: torch.zeros(n, dtype=torch.long, device=device)

                # 3) build inputs
                padded_window  = torch.cat([pad(pad_len),                window], dim=0)
                attention_mask = torch.cat([pad(pad_len), torch.ones(window.size(0), dtype=torch.long, device=device)], dim=0)
                pos_ids        = torch.cat([pad(pad_len), torch.arange(window.size(0), dtype=torch.long, device=device)], dim=0)

                windows.append(padded_window)
                attention_masks.append(attention_mask)
                position_ids_list.append(pos_ids)

            # stack and forward
            batch_tensor          = torch.stack(windows)
            attention_mask_tensor = torch.stack(attention_masks)
            position_ids_tensor   = torch.stack(position_ids_list)

            with torch.no_grad():
                outputs = self.model(
                    batch_tensor,
                    attention_mask=attention_mask_tensor,
                    position_ids=position_ids_tensor,
                    output_hidden_states=True
                )

            # extract the last token (position len(window)-1) from the chosen layer
            layer_acts = outputs.hidden_states[layer]  # (B, max_window_size, D)
            token_idxs = torch.tensor(
                [min(w.size(0), self.max_window_size) - 1 for w in windows],
                device=device
            )
            batch_embeds = layer_acts[torch.arange(len(windows), device=device), token_idxs, :]
            token_embeddings.append(batch_embeds.cpu())

            # free memory
            del batch_tensor, attention_mask_tensor, position_ids_tensor, outputs, layer_acts
            torch.cuda.empty_cache()

        return torch.cat(token_embeddings, dim=0).cpu()


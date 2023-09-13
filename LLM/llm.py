from typing import List, Dict, Optional
import torch.distributed as dist

from llama import Llama, Dialog

class LlamaChat:
    def __init__(self, 
                 ckpt_dir: str, 
                 tokenizer_path: str, 
                 max_seq_len: int, 
                 max_batch_size: int, 
                 temperature: float, 
                 top_p: float, 
                 max_gen_len: Optional[int] = None):
        print("Initializing distributed environment")
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=1)
        
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.active_dialog = []
        self.system_role_set = False
        
        self.generator = Llama.build(
            ckpt_dir=self.ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
        )
        
    def role_assignment(self, system_role: str):
        if not self.system_role_set:
            self.active_dialog.append({"role": "system", "content": system_role})
            self.system_role_set = True
            
    def clear_old_chat_history(self):
        self.active_dialog.clear()
        self.system_role_set = False
    
    def generate_response(self, user_input: str):
        self.active_dialog.append({"role": "user", "content": user_input})
        
        dialogs: List[Dialog] = [self.active_dialog]
        
        results = self.generator.chat_completion(
            dialogs,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        result = results[0]  # We only have one active dialog
        assistant_response = result['generation']
        
        # Append the generated response to active dialog
        self.active_dialog.append(assistant_response)
        
        # Show only the assistant's reply
        print(f"{assistant_response['role'].capitalize()}: {assistant_response['content']}\n")
        print("\n==================================\n")
        
    def show_text_history(self):
        for msg in self.active_dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print("\n==================================\n")

if __name__ == "__main__":
    llama_chat = LlamaChat(
        ckpt_dir="llama-7b_aitnowei/",
        tokenizer_path="tokenizer.model",
        max_seq_len=4000,
        max_batch_size=4,
        temperature=0.6,
        top_p=0.9
    )


import torch

class GPTDecoding(object):

    def decode(self,model,tokenizer,count,max_length=100):
        output_texts=[]
        c=0
        while c<count:
            encoding=model.generate(None,do_sample=True,max_length=max_length,top_p=0.8, top_k=50,
                                    pad_token_id=tokenizer.eos_token_id,num_return_sequences=10,
                                    eos_token_id=tokenizer.eos_token_id)
            batch_texts=tokenizer.batch_decode(encoding,skip_special_tokens=True)
            for text in batch_texts:
                if text.strip()!="":
                    output_texts.append(text)
                    c+=1

        return output_texts[:count]
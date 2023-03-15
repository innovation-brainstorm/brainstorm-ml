import torch
import torch.nn as nn
import numpy as np

class TopPDecoding(object):

    def decode(self,model,tokenizer,count,max_length=100,top_p=0.6):
        model.eval()

        
        output_texts=[]
        with torch.no_grad():

            for c in range(count):
                output_token_ids=[tokenizer.token_to_id("[BOS]")]
                hidden=None

                for i in range(max_length):

                    X=torch.zeros((1,1)).long()
                    X[0,0]=output_token_ids[-1]

                    if hidden is None:
                        output,hidden=model(X)
                    else:
                        output,hidden=model(X,*hidden)

                    prob=nn.functional.softmax(output)
                    
                    sorted_p,sorted_indices=torch.sort(prob,descending=False)
                    cumsum_p=sorted_p.cumsum(dim=-1)

                    mask=cumsum_p<=(1-top_p)
                    sorted_p=sorted_p.masked_fill_(mask,0)
                    sorted_p=np.asarray(sorted_p)
                    prob_v_norm=(sorted_p/sorted_p.sum()).ravel()

                    token_id=np.random.choice(sorted_indices.ravel(),p=prob_v_norm)


                    output_token_ids.append(token_id)

                    if token_id==tokenizer.token_to_id("[EOS]"):
                        break
                
                output_text=tokenizer.decode(output_token_ids)
                output_texts.append(output_text)

        return output_texts
        



import torch

class GreadySearchDecoder(object):

    def decode(self,model,tokenizer,max_length=100):
        model.eval()

        v=tokenizer.get_vocab_size()

        with torch.no_grad():
            output_token_ids=[tokenizer.token_to_id("[BOS]")]
            hidden=None

            for i in range(max_length):

                X=torch.zeros((1,1)).long()
                X[0,0]=output_token_ids[-1]

                if hidden is None:
                    output,hidden=model(X)
                else:
                    output,hidden=model(X,*hidden)

                token_id=output.argmax(1)

                output_token_ids.append(token_id)

                if token_id==tokenizer.token_to_id("[EOS]"):
                    break
            
            output_text=tokenizer.decode(output_token_ids)
        print(output_text)
        



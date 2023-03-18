


class GPTDecoding(object):

    def decode(self,model,tokenizer,count,max_length=100):

        output_texts=[]
        for i in range(count):
            encoding=model.generate(None,do_sample=True,max_length=max_length,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=1)
            output_text=tokenizer.batch_decode(encoding,skip_special_token=True)
            output_texts.append(output_text[0].strip())
        return output_texts
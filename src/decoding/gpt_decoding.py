


class GPTDecoding(object):

    def decode(self,model,tokenizer,count,max_length=100):

        encoding=model.generate(None,do_sample=True,max_length=max_length,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                num_return_sequences=count)
        output_text=tokenizer.batch_decode(encoding,skip_special_tokens=True)
        output_text=[text.strip() for text in output_text]
        return output_text
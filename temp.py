


train_data=TextDataset(root="../data/news_titles",split="train")

tokenizer=Tokenizer(BPE(unk_token="[UNK]",end_of_word_suffix="</w>"))

trainer=BpeTrainer(special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"],end_of_word_suffix="</w>")

tokenizer.prtokenizer=Whitespace()
tokenizer.decoder=decoders.BPEDecoder()

tokenizer.post_processor=TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[("[BOS],2"),("[EOS]",3)]
)

tokenizer.enable_padding()

tokenizer.train(["../data/news_title/train.txt"],trainer)

print(tokenizer.get_vocab_size())


lass CharGenerator(object):

    def __init__(self,model):
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer

    def train(self,dataloader,tokenizer):
        size=len(dataloader.dataset)
        print_loss=0
        running_loss=0

        for batch,texts in enumerate(dataloader):
            encodings=tokenizer(texts,truncation=True,padding="max_length",max_length=max_length)
            X=data_collator([encodings])


            output=model(**X)
            loss=output.loss

            print_loss+=loss
            running_loss+=loss



            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % print_every==0:
                print_loss_avg,current=print_loss if batch==0 else print_loss/print_every,batch*len(texts)
                print_loss=0
                print(f"loss:{print_loss_avg:>7f} [{current:>5d}/{size:>5d}]")
        
        return running_loss/len(dataloder)

    def test(self,dataloader):
        size=len(dataloader.dataset)
        total_loss=0
        correct=0
        sent_length=0

        with torch.no_grad():
            for batch,texts in enumerate(dataloader):

                encodings=tokenizer(texts,truncation=True,padding="max_length",max_length=max_length)
                X=data_collator([encodings])


                output=model(**X)
                total_loss+=output.loss

        total_loss/=len(dataloder)
        correct=0

        print(f"Test Error:\n Accuracy:{100*correct:>0.1f}%, Avg loss:{total_loss:>8f}\n")

        return total_loss

    def generate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


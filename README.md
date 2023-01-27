# Finetuned_DistilBERT
The [BERT](https://arxiv.org/abs/1810.04805) model which stands for Bidirectional Encoder Representations from Transformers was proposed by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It’s a bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia. 

From the paper's abstract : It can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

While [DistilBERT](https://arxiv.org/abs/1910.01108) is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.

In this notebook, I'll be finetuning the [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) pre-trained model on the IMDB Dataset for sentiment analysis.

## Requirements
- [PyTorch](https://pytorch.org/)
- Hugging Face's Transformers library
- IMDB dataset (included in the Hugging Face's Transformers library)
 
You can install Pytorch from its website. As for the rest two, you can just run the first cell of the notebook.

## About the trained model
The model was trained on 4000 samples from the IMDB Dataset. It gave the following metrics on evaluation:
```
{
  'eval_accuracy': 0.89,
  'eval_precision': 0.8461538461538461,
  'eval_recall': 0.9533333333333334,
  'eval_f1': 0.896551724137931
}
```
## Loading the saved model
The saved model can be downloaded from [here](https://drive.google.com/file/d/1R4oKxOX75dTivjrBMMpdClBReVzpHoQa/view?usp=sharing). Just use the Python code below to load it:
```
model = torch.load('finetuned_bert_model')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```
This tokenizer was used during training, so use the same one with the loaded model.

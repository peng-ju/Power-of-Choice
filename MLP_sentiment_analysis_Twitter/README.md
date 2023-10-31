### Dataset Sent140 is download from "http://help.sentiment140.com/for-students"

### Glove from "https://nlp.stanford.edu/projects/glove/"
- Pre-trained 200D GloVe embedding. 

### parameters applied for training
- batch size: b = 32
- learning rate: η = 0.05, without learning rate decay.
- number of clients: K = 314, each tweeter account is a client.
- number of selected clients for each communication round: m = 8
- local trainig update for each communication round: τ = 100
- The output labels 0 as positive, and 1 as negative. 
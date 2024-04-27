# **1. Sentiment Analysis**

Sentiment Analysis if a task done in NLP to find the sentiment of a given document of text. The idea is to find the user sentiment whether it is postive, negative, or neutral given the document written by him/her.

Sentiment Analysis will give us some important information about user intent, behaviour and how a product of service is performing among users. There are different methods in NLP to perform Sentiment Analysis, like we can build Neural Language Models to recognize sentiment, but one simple and efficient sentiment analysis algorithm is Naive Bayes, which model the probability of a document being in a particular class.

## **1.1 Naive Bayes for Sentiment Analysis**

Naive Bayes is a probabilistic model used to estimate the likelihood of a particular class or outcome based on input feature evidence. It is a popular and simple model for sentiment analysis, offering better performance with less computation compared to complex neural architectures for text classification tasks.

Naive Bayes assumes that the features of words are independent of each other, hence the term 'Naive'. It does not consider the relationship between words or features. Instead, it treats each word and its corresponding target individually to model the probability of that word occurring given the document or context.

The Naive Bayes algorithm is derived from the well-known **Bayes' theorem**, which uses conditional probabilities to understand how to update our beliefs in light of new evidence.

For sentiment analysis task, out main goal is to identify the sentiment class $c$ given a document $d$:

$$P(c|d)$$

For multiple classes, Naive bayes will give us probability for each of them given the document, for finding the right class we have to calculate the argmax of $P(c|d)$

$$\hat{c} = \arg \max_{c \in C} P(c|d)$$

For finding the probability of class given document, we have bayes theorem:

$$\hat{c} = \arg\max_{c \in C} P(c|d) = \arg\max_{c \in C} \frac{P(d|c)P(c)}{P(d)}$$

To simplyfy the equation, we can remove the denominator since $P(d)$ does not change for each class; it remains constant, out goal is to find the most likely class for same document d (Since we are dealing with one single document at a time to classify it to a class).

$$\hat{c} = \arg\max_{c \in C} P(c|d) = \arg\max_{c \in C} P(d|c)P(c)$$

* Here $P(d|c)$ is the likelyhood of a document to be in a particualr class
* $P(c)$ is the prior probability.

Document is not the entire document itself, it is set of features (words), so we can form the equation as:

$$
\hat{c} = \underset{c \in C}{\mathrm{argmax}} \ P(f_1, f_2, ..., f_n|c) \ P(c)
$$

Here Naive bayes assumes that each of the features are independed to each other and dependent to the class only. So we can find the probabilities for each feature given c independently:

$$P(f_1, f_2, ..., f_n|c) = P(f_1|c) \cdot P(f_2|c) \cdot \ldots \cdot P(f_n|c)$$

So the general term will be:

$$\hat{c}_{\text{NB}} = \arg\max_{c \in C} P(c) \times \prod_{f \in F} P(f | c)$$

To simply computation and modeling probabilities, we usually do Naive bayes in log space,

$$\hat{c}_{\text{NB}} = \arg\max_{c \in C} \log P(c) + \sum_{i \in \text{positions}} \log P(w_i | c)$$

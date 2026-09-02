# my-first-project
SmartFix Hawassa is an intelligent diagnostic assistant designed to help local businesses and technicians troubleshoot printer, copier, and network infrastructure errors instantly. By leveraging natural language processing and classification algorithms, the tool translates error symptoms or codes into precise, step-by-step repair guides.

Conversation with Gemini
So far we’ve been working with numerical data: when predicting prices of cabins (or "mökki" in Finland) we’ve looked at the size of the cabin, the number of indoor toilets and so on. All of these qualities have been expressible as numbers.

But what if we also had access to textual descriptions of the cabins? For instance, there could be reviews from former tenants: “beautiful lake-side location with lots of nature around!”, or “the cabin is right under a flight path, very noisy!” This may be valuable information, but how can we use it?

Giving text a representation that computers can work with is the foundation of Natural Language Processing (NLP). NLP comprises techniques that enable us to solve various tasks such as internet search, document categorization, and automatic question answering (think of smart assistants such as Siri and Alexa).

By and large, computers are number-crunching machines. So, in order to handle text data, we need to find a way to encode it as numbers. One way to do this is to use the “bag of words” approach, where we just count the number of occurrences of each word in a document.

Below we show a simple example of a bag-of-words matrix where each row describes a line out of a nursery rhyme. Anyone familiar with English language rhymes can probably identify it as the This Little Piggy.The name “bag of words” comes from the fact that the order in which the words are written is not used in any way, and only the words and the number of times they appear matter. It is as if all the words in the sentence are all jumbled together in a big bag so that the order in which they were put into the bag is completely lost. The motivation for ignoring the order of the words is that the number of possible words is already huge and it is challenging enough to learn models that involve thousands of parameters. Models where the occurrence of "this little Piggy" means something else than "Piggy little this" tend to require even more parameters. So we'll start with a simple bag-of-words approach.

You may recall the spam email filter from the Introduction to AI (the first part of the Elements of AI): it was based on the naive Bayes method and involved multiplying likelihood ratios corresponding to the words in the message. You can think of it as a bag-of-words technique because if you do the math in the naive Bayes fashion, the outcome is the same no matter which order the words are given.

A simple example of using the bag-of-words representation would be a similarity comparison, which we could use, for example, in the nearest neighbor method to determine the Manhattan distance between any two lines of text.Exercise 17: Bag of wordsConsider the table above that contains the bag-of-word representation of the This Little Piggy nursery rhyme. Compare the lines word by word and add up the differences in the counts of each word. For example the difference (distance) between the first two lines becomes

|1–1| + |0–1| + |0–0| + |1–0| + ... + |1–1| = 0 + 1 + 0 + 1 + ... + 0 = 5

where the |·| marks the absolute value so for example |0–1| = 1.

Which two lines are the most similar to each other?

To find the most similar lines, calculate the Manhattan distance (the sum of absolute differences in word counts) for each pair.

Comparing the 3rd line ("This little pig had roast beef") and the 4th line ("This little pig had none"):

this: ∣1−1∣=0

home: ∣0−0∣=0

way: ∣0−0∣=0

to: ∣0−0∣=0

cried: ∣0−0∣=0

wee: ∣0−0∣=0

went: ∣0−0∣=0

all: ∣0−0∣=0

piggy: ∣1−1∣=0

stayed: ∣0−0∣=0

had: ∣1−1∣=0

none: ∣0−1∣=1

beef: ∣1−0∣=1

and: ∣0−0∣=0

the: ∣0−0∣=0

roast: ∣1−0∣=1

market: ∣0−0∣=0

little: ∣1−1∣=0

Summing these absolute differences yields a total distance of 3, which is the lowest distance among all pairs.

Correct Answer:

the 3rd and the 4th

One problem with the bag-of-words approach is that we are basically placing equal importance on any word: ten occurrences of the word “a” is just as important as ten occurrences of the word “magnificent.” But if we were, for example, considering text sentiment (analyzing how someone feels), “a” and “magnificent” are obviously not equally important.

One popular and effective solution that we can apply to improve on the simple bag-of-words representation is called by the acronym tf-idf.The technique called by the cumbersome name Term Frequency Inverse Document Frequency (tf-idf) places more weight on occurrences of infrequent words compared to common words like 'a', 'the', 'is', and so on.In its simplest form the algorithm is as follows:

Calculate the frequency (the number of occurrences divided by document length) for each word in your collection of documents. This is the “term frequency”, or

𝑡

𝑓

tf. (Note: Ignore punctuation and capitalization when doing this.)

Calculate how many documents each word appears in, and divide this by the total number of documents. This is the “document frequency”, or

𝑑

𝑓

df. Since we wish to assign less weight to common words, we will use the inverse of this,

1

÷

𝑑

𝑓

1÷df.

There are different ways to combine these two numbers to assign weights to each word. The most common is the product of the term frequency and the logarithm of the inverse of the document frequency:

𝑡

𝑓

−

𝑖

𝑑

𝑓

𝑡

𝑓

×

𝑙

𝑜

𝑔

(

1

÷

𝑑

𝑓

)

tf−idf=tf×log(1÷df).

This may sound complicated, but don’t panic! It will make more sense with the example below.

Let’s consider these three sentences:

"He really, really loves coffee"

"My sister dislikes coffee"

"My sister loves tea"

In the tf–idf terminology, each of the sentences is a document and collectively they form our corpus. To calculate the tf-idf representation of each document, we start by listing all the unique words we have in the corpus. In this case they are: 'coffee', 'dislikes', 'he', 'loves', 'my', 'really', 'sister', and 'tea'.

Next we look at each document in turn, count the occurrences of each word in it, and divide the number by the length of the document. Remember: frequency is the rate of occurrence, so the number of occurrences divided by the total number of words in a document:

Document 1: he:

1

5

5

1

​

, really:

2

5

5

2

​

, loves:

1

5

5

1

​

, coffee:

1

5

5

1

​

Document 2: my:

1

4

4

1

​

, sister:

1

4

4

1

​

, dislikes:

1

4

4

1

​

, coffee:

1

4

4

1

​

Document 3: my:

1

4

4

1

​

, sister:

1

4

4

1

​

, loves:

1

4

4

1

​

, tea:

1

4

4

1

​

Halfway there! Next, let’s calculate the document frequencies of each word. The document frequency of a word is the number of documents that contain at least one occurrence of the word:

coffee:

2

3

3

2

​

, dislikes:

1

3

3

1

​

, he:

1

3

3

1

​

, loves:

2

3

3

2

​

, my:

2

3

3

2

​

, really:

1

3

3

1

​

, sister:

2

3

3

2

​

, tea:

1

3

3

1

​

Now all there is left is to calculate for each word in each document its tf-idf score by multiplying the term frequency with the logarithm of the inverse of the document frequency.

You should try calculating these numbers by yourself. To get the logarithms, you'll either need a good old fashioned desk calculator with a 'log' key or a spreadsheet application (just type for example '=log(3)' to get the value

0.4771212547

0.4771212547). You can even do the math in many web browsers by simply typing the formula in the search bar.

The base of the logarithm, is not very important, but to get the same numbers as in our examples and exercises, you should use the base-10 logarithm. This seems to be the default in many spreadsheet applications and browsers, but not in desktop calculators or programming languages (such as Python). If you use Python, you can get the base-10 logarithm of x by using the function math.log(x, 10).

Take for example the word “he” in Document 1. The term frequency is

1

5

5

1

​

since the word appears once, and the document frequency is

1

3

3

1

​

since the word appears in one out of the three documents. Summa summarum, the tf-idf value of 'he' in Document 1 is given by

1

5

×

𝑙

𝑜

𝑔

(

3

1

)

5

1

​

×log(

1

3

​

), which equals approximately

0.095

0.095.

Here are all the tf-idf calculations for all the words in the first document:The logic behind tf-idf is sound: if a word is very common across all documents (for example “the”, “is”, “a”, “and” etc), it is not very informative when describing a document, even if it appears many times in the document. On the other hand, if a word is very rare in the corpus (“epigenetics”, “gradient”), even a single appearance in a document might be informative.Exercise 18: TF-IDFLet’s use Humpty Dumpty as our corpus:

Humpty Dumpty sat on a wall,

Humpty Dumpty had a great fall.

All the king's horses and all the king's men

Couldn't put Humpty together again.

(Remember, you can type the tf-idf equation into a search engine on your browser to do the calculation)

What is the term frequency for the word "Humpty" in line 1 of Humpty Dumpty?

1/6 (since "Humpty" appears 1 time out of 6 total words in the first line).

What is the term frequency for the word "all" in line 3?

2/9 (since "all" appears 2 times out of 9 total words in the third line).

What is the document frequency for the word "Humpty"?

3/4 (since "Humpty" appears in 3 out of the 4 total lines/documents).

What is the tf-idf score for word "Humpty" in line 4 of Humpty Dumpty?

~0.02 (calculated as  
5
1
​
 ×log 
10
​
 ( 
3
4
​
 )≈0.2×0.1249≈0.025).

OverfittingAlrighty. We have now reached a point where we understand at least some machine learning techniques, and you're probably eager to try them out. Let's go and build some cool... ALARM ALARM ALARM! What? A warning siren starts blaring with a sign 'Fasten your overfitting safety belt'. What the heck is going on?

Before we go about building any cool machine learning applications, we need to have a serious discussion about something called overfitting. Overfitting is one of the fundamental issues that should be a mandatory part of any practical machine learning course. Without understanding it, one can do a great deal of harm. So what is it?

Key terminology

Overfitting

Simply put, overfitting means being too confident in predictions that worked in the training data. We mentioned the distinction between training data and test data in passing above when discussing linear regression and cabin prices. A linear model with five predictor variables (like cabin size, the size of the sauna, the distance to a lake etc.) adjusted to predict the prices in training data including three cabins will dutifully replicate the three prices exactly. The same applies when there are 10 predictor variables and 10 cabins, and in any situation where the number of predictor variables is the same as the sample size. Still, it is clear that the perfect 'predictions' in the training data aren't a guarantee of perfect predictions for any other data. This, in essence, is an extreme case of overfitting.

Another equally extreme case of overfitting occurs with the nearest neighbor method. Recall the cabin pricing problem again. We have some training data on cabin sales where we know the cabin details and the paid price. Let's see how the nearest neighbor method predicts the price of any one of them. Suppose one of the cabins is the first one in our example data above: 66 sq. meters, a 5 sq. meter sauna, 15 meters from a lake, with 2 indoor bathrooms, and 500 meters away from neighbors, with a price tag of €258,250. Note that this cabin is our "test data" point but it is also included in the training data. If we feed the cabin details back into the nearest neighbor method, it will simply find the exact same cabin in the training data and determine that it is the nearest neighbor of the test data point. Therefore, it will predict the price as €258,250. Since this was the price in the test data, we find that the price is predicted exactly. The same goes for any other cabin in the training data.

As you noticed, when we added more cabins in the training data than there were predictors (in our case five), the linear model could no longer fit the training data perfectly. In other words, we would say that the training error is not zero: this happens virtually always when the number of predictor variables is less than the sample size for any linear model (unless the data happen to be very special). Non-linear models, which we'll study in the next chapter, allow more flexibility but we may still get a very small training error even if the sample size is larger than the number of predictor variables.

The most important observation to make here is that a small training error, especially if it is obtained by fitting a complex, possibly non-linear model to a small training data set, doesn't guarantee that the model actually predicts new data well. In fact, a zero training error like in the case of our linear model or the nearest neighbor method, may still be followed by very poor prediction accuracy on test data that doesn't overlap with the training data.

So we've established that overfitting is not a good thing. How do we then avoid it? The first line of defense is something that some might already be familiar with from the exercises: splitting your data into training and testing data. When you train the model with one part of your original data and test its performance with another, you will have at least some idea of how well your model generalises when using unseen data. You might still have lingering doubts: was the split done in a good way? Would the performance of the model be radically different if the data were split in another way? One simple solution to this is to split the data into n different sets, and train the model n times – each time with a different combination of n - 1 sets, with the remaining set being used as a test set. This way you will get n estimates on how your selected model performs when using unseen data. This is called leave-one-out cross-validation and it is one of the simplest ways of doing cross validation.

Since overfitting is such a pain in the neck in the realm of machine learning, many people have spent time and energy to come up with ways to combat it. Most of these will be out of the scope of this course, but interested readers can look up methods like regularisation and dropout, both of which are examples of widely understood and used methods in both linear and logistic regression and neural networks.Exercise 19: Looking out for overfittingLet’s for a moment imagine you have a data set on 1000 email messages labeled as either spam or not. Out of the 1000 messages, 990 are legitimate emails, and 10 are spam.

Then you split your data into training and test sets in such a way that both labels are present in both sets in equal ratios, and then train a classifier on the data.

What would you set as the baseline accuracy that your model has to outperform in order to be considered worthwhile?

99% accuracy (classify everything as the majority class)

Logistic regressionOur next topic, neural networks, tends to attract more interest than the other topics on this course combined. Perhaps this is explained by the hope to understand our own mind, which – as far as it is understood – emerges from neural processing in our brain.

We will, however, start from an idea that connects traditional regression modeling to neural networks, namely logistic regression. This will help us explore the machine learning jungle without losing our bearings and ending up with a handful of isolated topics. In addition, logistic regression is actually a very useful and practical technique.

The starting point is the familiar linear regression model:

𝑎

+

𝑐

1

×

𝑥

1

+

.

.

.

+

𝑐

𝑝

×

𝑥

𝑝

a+c

1

​

×x

1

​

+...+c

p

​

×x

p

​

, which stands for a model where the number of coefficients and inputs is

𝑝

p so that the last term is some

𝑐

𝑝

×

𝑥

𝑝

c

p

​

×x

p

​

, and the three dots "..." is a short-hand notation that replaces the terms between the first and the last. Recall that the first term,

𝑎

a, is called the intercept parameter.

However, the difference between linear regression and logistic regression lies in what happens after the above linear function has been calculated. In linear regression, we're done and the linear function is the output. In logistic regression, we take the result of the linear function and apply a special non-linear function called the sigmoid function to it.The above chart summarizes the difference between linear regression and logistic regression: Both start by computing a linear function of the inputs, denoted by the Greek letter

∑

∑ (capital sigma) which is often used to denote addition. In logistic regression, the result is then passed through the sigmoid function.

The mathematical formula for the sigmoid function (see below) may look a bit intimidating but it serves a simple purpose: it converts any number to value between

0

0 and

1



The larger the input to the sigmoid function, the closer the output is to

1.0

1.0, and the smaller the input, the closer the output is to

0.0

0.0. For example, the value

0.0

0.0 is converted into

0.5

0.5 whereas the value

10.0

10.0 is converted into

0.9999546

0.9999546, which is a tiny bit short of

1.0

1.0. Likewise, the value

−

10

−10 is converted into

0.0000454

0.0000454 which is the same tiny bit greater than

0.0

0.0.

Since the output value is a number between

0

0 and

1

1, it can be interpreted as a probability. This is useful because we will use logistic regression to predict the probability of something happening, instead of using a numerical value (such as a price in euros or the amount of carbon emissions in kilograms). If we use logistic regression in medical diagnosis, the output of the model can be for example the probability of a certain disease. Or if we are trying to tell whether a smiley face is happy or not, the output of the model can be the probability that it's happy. When there are exactly two alternatives (happy or not, disease or no disease), the probability of the second outcome is of course one minus the probability of the first.

The model can then be used to classify different items into two classes so that we read the output from the model and interpret it as the probability of one of the classes. We often label the classes as

0

0 and

1

1, and let the output of the model be the probability of class

1

1.

The mathematical formula that defines the sigmoid function is as follows:



𝑠

(

𝑧

)

1

÷

(

1

+

exp

⁡

(

−

𝑧

)

)

s(z)=1÷(1+exp(−z))

The function

𝑒

𝑥

𝑝

exp denotes the exponential function

𝑒

𝑥

𝑝

(

𝑥

)

𝑒

𝑥

exp(x)=e

x

, (

𝑒

e raised to the power

𝑥

x) where

𝑒

e is the so- called Euler's number that is approximately equal to

2.718

2.718. It can be computed by the Python function exp in module math.What if we have more than two classes?

You may wonder whether the logistic regression model is only applicable to two-class problems where the output can be represented as zeros or ones. The good news is: it isn't. There is a simple generalization of the sigmoid function to multiple class labels, which is called softmax. We won't discuss it here but it's one more term to remember since it is heavily used in neural networks.Now that we understand the logistic regression model, which is just the linear regression model with the sigmoid function added at the tail end, it's time to discuss how such models are learned from data.

Just like in linear regression, the learning process involves adaptation of the weights or coefficients so that the model fits the training data as well as possible. In the case of standard linear regression, the optimization criterion is the squared error, and the least squares method gives us the solution in no time. However, in logistic regression, the outputs are probabilities and the observed responses are binary "yes/no" labels, and the optimization criterion is something other than the squared error.

The actual optimization criterion in logistic regression is called logarithmic loss. It is calculated by using the logistic regression model to evaluate the probability of the observed label: if the observed label is class

1

1, the probability is directly obtained from the sigmoid output; if the observed label is class

0

0, the probability is obtained from one minus the sigmoid output. The logarithmic loss for a given set of data is the sum of the negative logarithms of these probabilities. We will not go into the details of calculating the logarithmic loss nor will we discuss algorithms for minimizing it: we can skip this and go straight to the fun part, namely using ready-made tools that require merely a few Python commands and the job gets done!Exercise 20: Logistic regressionHere we have a (fictional) graph of data about hours spent learning Python vs the chance of getting a raise within a year. If your friend has spent 70 hours learning Python, what are her chances of getting a raise within a year?

Hours Studied: 70 hours.

Graph Evaluation: Locating 70 on the horizontal "Hours studied Python" axis and following it vertically to the sigmoid curve reveals a probability of approximately 90% on the vertical axis.

Comparison: Because 90% exceeds the 80% threshold, the probability falls into the highest bracket provided.

Correct Answer:

at least 80%

From logistic regression to neural networksNow that we have studied logistic regression, the basic idea of neural networks is actually only a small step away. We keep the idea of a linear function combined with a non-linear function such as the sigmoid function. What neural networks add to this is that we connect multiple such models together to obtain a network.

When talking about neural networks, we use slightly different terminology. Instead of coefficients we say weights. The non-linear part of the model, which in the case of logistic regression was the sigmoid function, is called the activation function. One such model is called a neuron. The neurons are connected to each other by letting the output of one neuron be an input of another.

Below is a diagram of a neural network. While it may look complicated at first glance, we will soon see it only contains parts that we are already familiar with, just arranged in a way that is unfamiliar to us at the moment.Let’s start breaking the diagram down. Our neural network in this case is a network of three input nodes

𝑥

1

,

𝑥

2

,

𝑥

3

x

1

​

,x

2

​

,x

3

​

forming the input layer, a hidden layer with two nodes

ℎ

1

,

ℎ

2

h

1

​

,h

2

​

and a bias node (more on this later), and an output layer with one node

𝑜

o. The input layer and the hidden layer are connected so that each node in the hidden layer behaves not unlike a logistic regression model where the input of the model comes from the input nodes. Likewise, the nodes in the hidden layer are connected to the output node in very much the same fashion: this time the nodes in the hidden layer provide the inputs to the output node.

The input layer is what it sounds like it is: it is your input data. The only new thing here is thinking about it in terms of nodes, where one node corresponds to one element in your input. For example in the cabin price case you could have

𝑥

1

x

1

​

be the cabin size,

𝑥

2

x

2

​

the size of the sauna and so on.

As we said, the nodes behave very similarly to a logistic regression model. Inside each node we calculate a linear combination of the inputs of the node – recall that in the case of the output node these are provided by the nodes in the hidden layer – and apply something like the sigmoid function to determine the actual output. As mentioned above, we call the function applied in the latter stage the activation function. This term is borrowed directly from neuroscience where neurons communicate by sending electrical pulses to other neurons when activated by received stimuli.

In the hidden layer we also have a bias node which is not connected to the input nodes. The purpose of the bias node is functionally the same as with the intercept term in a linear regression: it can shift the input coming from a layer to another layer by some constant value. In the network above it shifts the input the final output layer gets by a constant value determined by the weight of the coefficient to the output node. Bias nodes are not a mandatory feature of neural networks but are usually helpful in model performance.

What is the point in all this? If we step back and look at the neural network, it's a box where we insert the inputs and out comes the output. Looking at it as just a function with inputs and an output, it serves the same purpose as any regression or classification model. And since the individual neurons are not more complicated than a logistic regression model, how is this useful at all?



The magic happens at the point where the neurons inside the network use non-linear activation functions. Indeed, if we only use a linear activation function, such as the identity function

𝑓

(

𝑥

)

𝑥

f(x)=x, it can be shown that the whole network is just a big fat linear regression model. But if we use non-linear activations such as the sigmoid function or something called a rectified linear unit – just say ReLu to sound professional – the model suddenly becomes much more powerful than linear models. In fact, it becomes so powerful that with enough nodes in the network, we can learn to fit virtually anything data perfectly. A technical way to express this is to say that neural networks are "universal function approximators".

Even a relatively simple model with few neurons can learn things that are too much for linear models such as linear regression, logistic regression, or Naive Bayes.Exercise 21: Neural networks

How many input nodes does it have?

3 (as shown by the three yellow input nodes: 1.3, -2.2, and 9.5).

What about hidden nodes?

4 (as shown by the four red hidden nodes: a1, a2, a3, and a4).

What is the value of the linear combination in node a_1 before an activation function is applied to it?

3.848

Calculation: (1.3×−0.76)+(−2.2×0.22)+(9.5×0.56)=−0.988−0.484+5.32=3.848
If we choose to use the identity function as the activation function, what would be the output of this node?

3.848 (since the identity function outputs the linear combination value directly without modification).

If your final task would be to predict based on the height, weight, and length if a thing is a dog or a cat, what would you choose to be the activation function for the output layer?  

sigmoid, since the output will be a probability of it being one of them (ideal for binary classification).  

If we choose to use a sigmoid activation function, what would be the output of this node?

~0.98

Calculation: σ(3.848)= 
1+e 
−3.848
 
1
​
 ≈0.98
Your AI idea

The optional final task of this course consists of your own AI idea. We are not giving you a made-up problem to solve. Instead, we want to hear what kind of a problem you'd like to solve using AI – and how.So far, you have worked within the safe harbor of this course. The goal was to let you focus on the content instead of learning to install and use new applications and systems. The content was structured as small nuggets that you could solve one at a time to eventually construct the bigger picture – not unlike a jigsaw puzzle.

But real-world problems aren't always served in single-serve portions. It's also necessary to practice the art of identifying a specific problem and sketching a solution without having a ready-made problem statement to begin with. Building AI solutions for real-world problems also requires that we venture out of our safe harbor and engage with the world. This can mean actively seeking more information about a problem, talking to those affected by it, teaming up with the rest of the AI community, and learning to use new tools.

You can browse example projects created by others to gain inspiration, but don't hesitate to be creative and to think outside the box. The best-case scenario is that, with time, your project will grow into something that creates real change in your own community.It's time for your AI idea

To make it easier for you, we’re proposing you structure the project description around a list of topics. Once you have written down a few thoughts about each of these topics, you already have enough material to submit your project! If you’re up to it, you can also expand this into a working demo or prototype with code and data.The topics we’ll ask you to elaborate are:

Your idea in a nutshell: Name your project and prepare to describe it briefly.

Background: What is the problem your idea will solve? How common or frequent is this problem? What is your personal motivation? Why is this topic important or interesting?

Data and AI techniques: What data sources does your project depend on? Almost all AI solutions depend on some data. The availability and quality of the data are essential. Which AI techniques do you think will be helpful? Depending on whether you've been doing the programming exercises or not, you may choose to include a concrete demo implemented by coding, using some actual data!

How is it used: What is the context in which your solution is used, and by whom? Who are the people affected by it? It’s important to appreciate the viewpoints of all those affected.

Challenges: What does your project not solve? It’s important to understand that any technological solution will have its limitations.

What next: How could your project grow and become something even more?

Acknowledgments: If you’re using open source code or documents in your project, make sure you give credit to the creators. Mention your sources of inspiration, too.To complete this final project and to earn final project honors on your certificate, you should submit your idea formulated according to these questions as a README-file on GitHub. If this sounds new to you, don’t worry, we’ll guide you through the steps.

You can use this template whenever you want to craft and communicate your AI idea to anyone. If you already know who to send the idea to in order to take it forward, we encourage you to do so – you don’t have to submit an idea to pass this course. But if you feel you want to share something fun and useful with the world and find people to collaborate with to make your AI idea a reality, read on.How it works

Your AI idea project will not be graded by us (but if you submit an idea this will be mentioned on your certificate of completion for the course). You should think of it as something that you can do for yourself and your own community, not for us. It will also serve as a demonstration of your skills and your creativity. So even if the project is not required by us, we think it's an integral part of the Elements of AI journey: this is your gateway to the AI community.

Especially if you are doing your first AI project, you may feel intimidated and worried that it will turn out to be less than awesome. Don't worry, you can always continue to improve it and start new, perhaps even better projects. Doing great projects takes years of practice, but you have to start one day. Why not make today that day?How is the final project made and shared?

We will be using GitHub as the platform to store and share your project.

We chose GitHub as a platform for the AI ideas because it has such a strong foothold in the AI community and in the coding profession in general. If you’re into coding, you most likely already know this and have an account, and if not, let us assure you that it will be useful for you to have one when continuing your journey. In this project, you'll be creating a GitHub repository. Don't worry, it's easy and it's something that everyone in the AI scene tends to do.Note!

Git is for collaboration

GitHub is an online service for hosting git-based version control for software projects. Git is a technology originally developed in 2005 by Linus Torvalds, who is also known as the creator of the Linux operating system. With Git, developer teams can coordinate their work, avoid conflicts between different project versions and track file changes. Learning to use GitHub's basic functionality can be beneficial no matter what your profession is. It may even be newsworthy.Your final project submission will be a README in your Git repository

The readme file (or README in all caps) is an established way to share the documentation of a software project for users and developers. In GitHub, it’s the first page that’s shown to anyone who visits a project. A good README contains a description of the project and a user guide.

For our purposes, it’s enough to create just the README page – a kind of a poster for a future project. If you make it nice and inviting, who knows if one day you’ll find just the right people to collaborate with to make it a reality!

For those who program: we’d love it if you include an experiment, a demo or a prototype with the actual code in your repository. It can be your original work, but don’t forget the lovely world of sharing and contributing with open source projects. No matter what kind of project you’re planning to get started with, checking out some interesting and fun experiments can be fuel for your imagination (some of these examples are even accompanied with open source code).Note!

Remember the copyright

You can only use other people’s code, images, or other assets in your projects if the original creator has given the permission to do so. You can of course ask for permission personally, but sometimes you don’t have to: many creators license their work as Open Source or Creative Commons when they want to allow reuse under specific conditions. Even if a project is shared with the world under such a license, you still need to respect the creators’ rights to their work: for example, in most cases, you need to mention who the creator is. Sharing is the cornerstone of modern, digital culture – we encourage you to contribute to this culture by licensing your own work, too.Create your project page

To get started, log in to GitHub. Create an account if you don't have one yet.

To create a new README, you need to create a GitHub repository first. The repository is the home to all your project files: text, code, images, sound – any assets needed. At minimum, you can create a text file and add a few images to beautify the project description.

Below is a GIF of how to create a repository:Instructions

Create a new repository from the ‘+’ menu and set up the project

Give your repository a short, unique name. The name can be changed afterwards from Settings if you don’t yet know what the topic of the project will be

The repository must be created as Public so you can share it as your Building AI final project

Check the box ‘Initialize this repository with a README’

If you want, you can learn about adding a license to your project (or change the setting later). With licenses you control how your project can be used by others

Your AI Idea: Project Framework
To complete the final project for the Elements of AI and earn honors on your certificate, you can structure your project description as a README.md file in a public GitHub repository.

Here is the recommended template and breakdown of the topics you should address to pitch your real-world AI solution effectively:

1. Your Idea in a Nutshell
Project Name: Give your project a clear, memorable, and descriptive name.

Elevator Pitch: Summarize what your AI solution does, the core technology it uses, and the primary benefit it brings in 2–3 sentences.

2. Background
The Problem: What specific real-world problem are you trying to solve? How frequently does it occur?

Personal Motivation: Why do you care about this problem? What inspired you to tackle it?

Significance: Why is solving this important for your community or industry?

3. Data and AI Techniques
Data Sources: Where will the data come from? (e.g., public datasets, web scraping, user-generated logs, sensor data). Discuss data quality, privacy considerations, and availability.

AI/ML Techniques: Which models or algorithms fit best? (e.g., Linear/Logistic Regression, Naive Bayes for text classification, or a Multi-Layer Neural Network).

Demo/Prototype: If you code, mention any scripts, notebooks, or small experiments included in your repository using sample data.

4. How It Is Used
User Context: In what exact setting will your application be used? Who is the end user?

Impacted Parties: Who else might be affected—either positively or negatively—by your AI system? Address ethical considerations and user trust.

5. Challenges
Limitations: What does your project not solve?

Potential Pitfalls: Discuss risks like overfitting, bias in the training data, edge cases, or lack of computational resources.

6. What Next?
Future Roadmap: How could this project scale? What features would you add if you had more time, collaborators, or funding?

7. Acknowledgments
Credits: List any open-source code libraries, frameworks (like Scikit-Learn or TensorFlow), datasets, or tutorials that inspired or served as building blocks for your work.

Step-by-Step Guide to Publishing on GitHub
Log in to GitHub: Go to GitHub and sign into your account (or create a new one).

Create a Repository: Click the + icon in the top right corner and select New repository.

Configure Settings:

Give your repository a unique name (e.g., ai-cabin-price-predictor).

Set the visibility to Public (required so others can view your final project).

Check the box to Initialize this repository with a README.

Edit your README: Click on the README.md file inside your new repository, click the pencil/edit icon, paste your project details using the structure above, and click Commit changes.

What specific problem or domain are you thinking about focusing on for your AI idea?

Create the README project plan

Now it’s time to draft your project plan into a README file. The README in GitHub is formatted using markdown, which is a common way to add styling to text. Markdown is used on many online platforms.

To make things simple, we’ve created a template you can use to draft your README:

SmartFix Hawassa: AI-Powered Office Equipment Diagnostics
Your Idea in a Nutshell
SmartFix Hawassa is an intelligent diagnostic assistant designed to help local businesses and technicians troubleshoot printer, copier, and network infrastructure errors instantly. By leveraging natural language processing and classification algorithms, the tool translates error symptoms or codes into precise, step-by-step repair guides.

Background
Small businesses, schools, and offices frequently suffer from operational downtime caused by sudden office equipment malfunctions and network failures. Currently, diagnosing these issues requires manually searching through dense user manuals or waiting for an on-site technician. This project stems from firsthand experience in local IT support and maintenance, aiming to bridge the gap between technical expertise and rapid problem resolution in regional offices.

Data and AI Techniques

Data Sources: Compiled datasets of standard error codes, hardware logs, troubleshooting manuals, and frequently occurring maintenance queries.

AI/ML Models: A text classification model (such as Naive Bayes or a multi-layer perceptron) trained to categorize user-submitted error descriptions and map them to verified root causes and solutions.

Prototype: A lightweight Python script utilizing scikit-learn to process symptom text inputs and output ranked troubleshooting recommendations.

How It Is Used
The solution is deployed as a simple web interface or a Telegram-based chatbot. Office managers or field technicians type in the error code or a brief description of the hardware fault (e.g., "printer paper jam error 50.3" or "router blinking red light"), and the system instantly returns corrective actions, minimizing equipment downtime.

Challenges
The tool acts strictly as a diagnostic advisor and cannot perform physical repairs, replace hardware components, or fix physical wiring faults. Its accuracy depends heavily on the clarity of the user's input description and the completeness of the error-code database.

What Next
Future developments will introduce computer-vision capabilities, allowing users to upload a photo of a blinking indicator light or physical error screen for automated visual diagnostics. Multilingual support incorporating Amharic alongside English will also be expanded to maximize accessibility for local users.

Acknowledgments
Inspired by everyday IT service challenges managed through local technical solutions in Hawassa. Built utilizing open-source Python data science libraries including Pandas and Scikit-Learn.

make this project easier to copy github

Markdown
# SmartFix Hawassa: AI-Powered Office Equipment Diagnostics

## Your Idea in a Nutshell
SmartFix Hawassa is an intelligent diagnostic assistant designed to help local businesses and technicians troubleshoot printer, copier, and network infrastructure errors instantly. By leveraging natural language processing and classification algorithms, the tool translates error symptoms or codes into precise, step-by-step repair guides.

## Background
Small businesses, schools, and offices frequently suffer from operational downtime caused by sudden office equipment malfunctions and network failures. Currently, diagnosing these issues requires manually searching through dense user manuals or waiting for an on-site technician. This project stems from firsthand experience in local IT support and maintenance, aiming to bridge the gap between technical expertise and rapid problem resolution in regional offices.

## Data and AI Techniques
* **Data Sources:** Compiled datasets of standard error codes, hardware logs, troubleshooting manuals, and frequently occurring maintenance queries.
* **AI/ML Models:** A text classification model (such as Naive Bayes or a multi-layer perceptron) trained to categorize user-submitted error descriptions and map them to verified root causes and solutions.
* **Prototype:** A lightweight Python script utilizing `scikit-learn` to process symptom text inputs and output ranked troubleshooting recommendations.

## How It Is Used
The solution is deployed as a simple web interface or a Telegram-based chatbot. Office managers or field technicians type in the error code or a brief description of the hardware fault (e.g., "printer paper jam error 50.3" or "router blinking red light"), and the system instantly returns corrective actions, minimizing equipment downtime.

## Challenges
The tool acts strictly as a diagnostic advisor and cannot perform physical repairs, replace hardware components, or fix physical wiring faults. Its accuracy depends heavily on the clarity of the user's input description and the completeness of the error-code database.

## What Next
Future developments will introduce computer-vision capabilities, allowing users to upload a photo of a blinking indicator light or physical error screen for automated visual diagnostics. Multilingual support incorporating Amharic alongside English will also be expanded to maximize accessibility for local users.

## Acknowledgments
Inspired by everyday IT service challenges managed through local technical solutions in Hawassa. Built utilizing open-source Python data science libraries including Pandas and Scikit-Learn.
Gemini is AI and can make mistakes.





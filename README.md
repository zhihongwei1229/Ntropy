# Ntropy

--------
TASK 1 (ML eng. only)
--------

Imagine the digits in the MNIST dataset (http://yann.lecun.com/exdb/mnist/) got
cut in half vertically and shuffled around. Although the
two datasets do not have any shared pixels, there are similarities in their
structure. Implement a way to restore the original dataset from the two halves,
whilst maximising the overall matching accuracy.



step 1,

I load MNIST data from Kersa


Step 2,

Created a simple CNN model by TensorFlow. Run this model on the the original dataset. This model's performance of Accuracy is +99% in train dataset and 98% in test dataset. (test.py)

Step 3,

split the MNIST data vertically. now train dataset is (120000, 28, 14), training the model against this splited dataset. the performance is 93.5% in train dataset and 93.2% in test dataset

Step 4,

I think if we mix left part and right part directly, these 2 training data may confuse the model. So the most important thing to improve the performance is find a way to tell the model whether the data is the left part or the right part of the original data. There are multiple way to do this, but I think the most easly way to do this is padding 0s to the missing part. 
For each row of left part,
imporved_dataset_row = [14 elements of left part] + [0]*14
For each row of right part,
imporved_dataset_row = [0]*14 + [14 elements of right part]

Run the model against the processed dataset. The profermance is 95% on train dataset and 93.94% in test data. (main.py)

Conclusion, save the splited dataset with padding can improve the overall accuracy.

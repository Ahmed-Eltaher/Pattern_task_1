Pattern_task_1

<h1>Code Overview</h1>

<h3># Online Algorithm</h3>
<p>
-This algorithm receives data set consists of (X,Y):
X  input values for each feature.
Y  classification of each training example (1,-1)
-The main objective to find the best possible W_omega “weight”, this is done by initializing w_omega with zeros and a variable “Delta” with zeros also both are same size as the number of features of the data.
-It iterates in a while loop until the delta value is very low that it won’t make any changes to the omega value.
-the while loop consists of for loop, iterates over the training set (inputs/training examples) each time it checks for the (w_omega * Xi) result and if it matches the true value of y (same sign  right classification), otherwise  wrong classification and means w_omega need to be updated.

<p>






<h3>Batch Perceptron Algorithm:
-This algorithm follows the same approach as the previous one.
•	Except that “Main Difference”: 
The updating of w_omega is not done for each training example, instead it’s done after the complete updating of Delta, and number of epochs is calculated for each time the omega is updated after n times we iterated over the total data set in updating delta.
</h3>






<h3>Problem 1</h3>
<p>1. Number of epochs needed to achieve almost zero error:

Online Algorithm: 

number of epoches for given delta mean graph = 50
Accuracy of test for this run = 100%

picture for delta graph:![](delta1.jpg)


Batch Algorithm:
number of epoches for given delta mean graph = 194

Accuracy of test for this run = 100%

picture for delta graph:![](delta2.jpg)

<p>




<h3>Problem 4</h3>
<p>1. Number of epochs needed to achieve almost zero error:

Online Algorithm: 

number of epoches for given delta mean graph = 4
Accuracy of test for this run = 50%

picture for delta graph:![](delta3.jpg)


Batch Algorithm:
number of epoches for given delta mean graph = 4

Accuracy of test for this run = 50%

picture for delta graph:![](delta4.jpg)

<p>





<h3>Classifiation model</h3>

<h4>test 1</h4>
<h4> Online Algorithm</h4>
<p> 
    Number of epoches for this run: 2

    Test accuracy: 100%
    
    picture of delta graph:
![](delta_m1.jpg)

    picture of classification model:
![](classification_m1.jpg)

<p>


<h4>test 2</h4>
<h4> batch Algorithm</h4>
<p> 
    Number of epoches for this run:  
     3

    Test accuracy: 100%
    
    picture of delta graph:
![](delta_m2.jpg)    

    picture of classification model:
![](classification_m2.jpg)    
    
<p>

<title>Linear Regression Basic Flow</title>
<h3 class="graf graf--h3">Defn</h3>
<ul>
	<li>ML is one of the applications of artificial intelligence (AI), that iteratively learn from the data, unlike typical programming.</li>
	<li>ML algos can find insights in the data even if they aren’t specifically instructed what to look for in the data.</li>
</ul>

<h3 class="graf graf--h3">Branches</h3>
<strong class="markup--strong markup--blockquote-strong">Supervised Learning</strong>: Uses Labeled data for prediction.(Eg: previous house sales with info how much each house sold for -- labeled dataset.), Once your model is ready we can use it for prediction where we only need features(size and floors of the house) of the unseen data to be given to the model.
<h4 class="graf graf--h4">    Types</h4>
<ul>
	<li><strong class="markup--strong markup--blockquote-strong">Classification</strong>: When we deal with categorical data, it is termed as classification. ex: Given person's Height and weight predict the gender. (predicting the class male vs female, Male and female are categories henceforth called categorical data.)</li>
	<li><strong class="markup--strong markup--blockquote-strong">Regression</strong>: When we deal with continuous data, it is termed as Regression.          ex: Given house size and no of floors predict the House selling price.(Predicting a continuous number of 52.5 lakh, 52.5 is a continuous data unlike)</li>
</ul>

<strong class="markup--strong markup--blockquote-strong">Unsupervised Learning</strong>: Unlabeled data, where we can <strong class="markup--strong markup--blockquote-strong">cluster </strong>data into similar groups by some common patterns among them. It's then up to the data-scientist how to interpret the clusters.

<strong class="markup--strong markup--blockquote-strong">Reinforcement learning</strong>: It works on trial and error methodology, example use cases are like a computer playing video games...

<hr />

<h2 class="graf graf--h3">                                       Math Essentials</h2>
<strong class="markup--strong markup--blockquote-strong">scalar vs vector: </strong><strong class="markup--strong markup--blockquote-strong"> </strong>Vector is something which got magnitude and direction, and a scalar helps to scale up the data.

<img class="alignnone  wp-image-15" src="https://themlrecipes.files.wordpress.com/2018/02/vector.png" alt="vector" width="130" height="141" />

ex: let's say a man walks at 1kmph(magnitude) in North East direction, which can be represented as [1, 1.5], now if we scale up the same vector by multiplying by 2 (magnitude/scalar), he reaches a point at [2, 3] after 2 hours.

A vector can be more than two dimensional, our computers are good at calculating Dot products of vector and hence in the following tutorial, we try to see a vectorized implementation for all mathematical equations.

<strong class="markup--strong markup--blockquote-strong">Line Equation:  </strong>y = mx + b,  aline equation which fits a linear relationship between x & y.

<hr />

<h2 class="graf graf--h3">                             <strong>Simple Linear regression</strong></h2>
<img class="" src="https://www.mathworks.com/help/symbolic/mupad_ug/math-statistics-fits-linear-36e42cfe.png" alt="Image result for linear regression" width="333" height="221" />

Let's say we have to fit a straight line to the given data, as shown above, the line equation can be as follows.

slope m = rise/run, (14-12)/(2-1) = 2

Y-intercept b = 9 (where line crosses y-axis, if we imagine X-axis starting from zero, Line would be crossing y-axis at y=9)

Y = 2X + 9

By this equation, we can predict any unseen point y on the graph given x, which let's say x=11, then we can predict Y = 31 by using the line equation.

Now if our points are little scattered unlike the previous example, we try to fit a better line of approximation which can do a decent job for all the given points.

<img src="https://scontent.fblr4-1.fna.fbcdn.net/v/t34.0-12/28534536_1403493129762387_1986054821_n.png?oh=414bfcee5200ea3d4507513230d645e8&oe=5A98DCB0" />

<strong class="markup--strong markup--blockquote-strong">Model:  </strong>Now based on above line equation we can approximate any seen or unseen y given x. This is something which we refer to the linear regression Model.

<strong class="markup--strong markup--blockquote-strong">Problem</strong>: Now the problem is to figure out a such a Straight line like in the above figure which can approximate by considering all the given points.

We have to make use of Loss function and gradient Descent to achieve the same.
<h2 class="graf graf--h3"><strong>  Cost Function</strong></h2>
Cost function $${J(\theta_0, \theta_1)}$$ tells us how well our model fits into the data. We can measure the accuracy of our hypothesis function by using a cost function. This takes an average difference of all the results of the hypothesis with inputs from x's and the actual output y's.
Average difference between the $${\hat{y}}$$ and y is known as <strong>Mean Squared Error</strong>.

<strong>Hypothesis Function:</strong> Rewrite the line equation y = mx + b as a function $${h_\theta(x) = \theta_0 +\theta_1x}$$ (Hypothesis Function) where $${\theta_0}$$=b and $${\theta_1}$$=m

$${\theta_i}$$ = weights(which scales the features), $${x_{i}}$$ = features, $${y_{i}}$$ = original output, $${\hat{y}_{i}}$$ = $${h_\theta(x)_i}$$ = expected output by hypothesis function


$${J(\theta_0, \theta_1) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left (h_\theta (x_{i}) - y_{i} \right)^2}$$

Now we have a way to measure our Hypothesis Function given $${\theta_0, \theta_1}$$, Next we have to automate finding best $${\theta_0, \theta_1}$$ which minimizes the cost function J, so that we get the best model to work with.

To automatically find best $${\theta_0}$$ and $${\theta_1}$$, we can use Gradient descent.

Gradient descent:  keep changing $${\theta_0}$$ and $${\theta_1}$$ to reduce $${J(\theta_0, \theta_1)}$$

<img class="" src=[a relative link](/assets/GD.gif) alt="Image result for linear regression" width="333" height="221" />


Intution:

eqn
$${\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)}$$

lets try to minimize only one parameter T1 for simplicity
$${
J(\theta) =\frac{1}{2m}
[\sum^m_{i=1}(h_\theta(x^{(i)}) -
y^{(i)})2 + \lambda\sum^n_{j=1}\theta^2_j
}$$

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>

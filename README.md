
# Personalised Cancer Diagnosis Results

# Bag of Words Tabular Result


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Naive Bayes','K' 'Nearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Machines', 'Random Forest [One Hot]', 'Random Forest [Response Coding]', 'Stacking [LR, SVM, NB] with LR', 'Stacking [LR,Linear,RF] with LR', 'Stacking [NB, LR, RF] with LR', 'Stacking [K-NN, NB, Linear] with LR', 'Stacking [LR, K-NN, Linear] with LR', 'Stacking [LR, Linear] with LR', 'Maximum Voting Classifier'],
    "Train (%)": [0.9,0.7,0.61,0.62,0.76,0.70,0.06,0.67,0.65,0.66,0.74,0.66,0.68,0.94],
    "CV (%)": [1.35,1.10,1.21,1.25,1.24,1.17,1.41,1.22,1.14,1.14,1.15,1.14,1.16,1.28],
    "Test (%)": [1.27,1.09,1.10,1.13,1.15,1.16,1.38,1.15,1.08,1.09,1.11,1.09,1.11,1.22],
    "Misclassification (%)": [44.36,39.66,38.34,37.21,39.47,38.53,51.87,37.29,35.33,35.33,34.58,36.09,36.39,38.19],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive Bayes</td>
      <td>0.90</td>
      <td>1.35</td>
      <td>1.27</td>
      <td>44.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>0.70</td>
      <td>1.10</td>
      <td>1.09</td>
      <td>39.66</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>0.61</td>
      <td>1.21</td>
      <td>1.10</td>
      <td>38.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>0.62</td>
      <td>1.25</td>
      <td>1.13</td>
      <td>37.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Machines</td>
      <td>0.76</td>
      <td>1.24</td>
      <td>1.15</td>
      <td>39.47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest [One Hot]</td>
      <td>0.70</td>
      <td>1.17</td>
      <td>1.16</td>
      <td>38.53</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest [Response Coding]</td>
      <td>0.06</td>
      <td>1.41</td>
      <td>1.38</td>
      <td>51.87</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR, SVM, NB] with LR</td>
      <td>0.67</td>
      <td>1.22</td>
      <td>1.15</td>
      <td>37.29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stacking [LR,Linear,RF] with LR</td>
      <td>0.65</td>
      <td>1.14</td>
      <td>1.08</td>
      <td>35.33</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stacking [NB, LR, RF] with LR</td>
      <td>0.66</td>
      <td>1.14</td>
      <td>1.09</td>
      <td>35.33</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stacking [K-NN, NB, Linear] with LR</td>
      <td>0.74</td>
      <td>1.15</td>
      <td>1.11</td>
      <td>34.58</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stacking [LR, K-NN, Linear] with LR</td>
      <td>0.66</td>
      <td>1.14</td>
      <td>1.09</td>
      <td>36.09</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Stacking [LR, Linear] with LR</td>
      <td>0.68</td>
      <td>1.16</td>
      <td>1.11</td>
      <td>36.39</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Maximum Voting Classifier</td>
      <td>0.94</td>
      <td>1.28</td>
      <td>1.22</td>
      <td>38.19</td>
    </tr>
  </tbody>
</table>
</div>



# Term Frequency - Inverse Word Frequency Tabular Result


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Naive Bayes','K' 'Nearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Machines', 'Random Forest [One Hot]', 'Random Forest [Response Coding]', 'Stacking [LR, SVM, NB] with LR', 'Stacking [LR,Linear,RF] with LR', 'Stacking [NB, LR, RF] with LR', 'Stacking [K-NN, NB, Linear] with LR', 'Stacking [LR, K-NN, Linear] with LR', 'Stacking [LR, Linear] with LR', 'Maximum Voting Classifier'],
    "Train (%)": [0.9,0.61,0.57,0.55,0.68,0.63,0.04,0.63,0.60,0.68,0.74,0.62,0.63,0.84],
    "CV (%)": [1.23,1.05,1.15,1.18,1.23,1.18,1.37,1.20,1.09,1.10,1.10,1.08,1.10,1.16],
    "Test (%)": [1.19,1.05,1.09,1.11,1.16,1.15,1.36,1.15,1.12,1.12,1.12,1.12,1.13,1.16],
    "Misclassification (%)": [43.23,35.90,37.59,36.65,39.28,41.54,57.70,37.74,36.39,35.93,36.09,36.39,36.39,38.79],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive Bayes</td>
      <td>0.90</td>
      <td>1.23</td>
      <td>1.19</td>
      <td>43.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>0.61</td>
      <td>1.05</td>
      <td>1.05</td>
      <td>35.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>0.57</td>
      <td>1.15</td>
      <td>1.09</td>
      <td>37.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>0.55</td>
      <td>1.18</td>
      <td>1.11</td>
      <td>36.65</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Machines</td>
      <td>0.68</td>
      <td>1.23</td>
      <td>1.16</td>
      <td>39.28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest [One Hot]</td>
      <td>0.63</td>
      <td>1.18</td>
      <td>1.15</td>
      <td>41.54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest [Response Coding]</td>
      <td>0.04</td>
      <td>1.37</td>
      <td>1.36</td>
      <td>57.70</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR, SVM, NB] with LR</td>
      <td>0.63</td>
      <td>1.20</td>
      <td>1.15</td>
      <td>37.74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stacking [LR,Linear,RF] with LR</td>
      <td>0.60</td>
      <td>1.09</td>
      <td>1.12</td>
      <td>36.39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stacking [NB, LR, RF] with LR</td>
      <td>0.68</td>
      <td>1.10</td>
      <td>1.12</td>
      <td>35.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stacking [K-NN, NB, Linear] with LR</td>
      <td>0.74</td>
      <td>1.10</td>
      <td>1.12</td>
      <td>36.09</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stacking [LR, K-NN, Linear] with LR</td>
      <td>0.62</td>
      <td>1.08</td>
      <td>1.12</td>
      <td>36.39</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Stacking [LR, Linear] with LR</td>
      <td>0.63</td>
      <td>1.10</td>
      <td>1.13</td>
      <td>36.39</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Maximum Voting Classifier</td>
      <td>0.84</td>
      <td>1.16</td>
      <td>1.16</td>
      <td>38.79</td>
    </tr>
  </tbody>
</table>
</div>



# TF-IDF with Top 1K Features


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Naive Bayes','K' 'Nearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Machines', 'Random Forest [One Hot]', 'Random Forest [Response Coding]', 'Stacking [LR, SVM, NB] with LR', 'Maximum Voting Classifier'],
    "Train (%)": [0.78,0.62,0.58,0.56,0.66,0.83,0.07,0.80,0.94],
    "CV (%)": [1.19,1.08,1.05,1.07,1.10,1.18,1.37,1.16,1.21],
    "Test (%)": [1.20,1.04,1.06,1.09,1.12,1.18,1.40,1.14,1.19],
    "Misclassification (%)": [38.15,37.59,37.03,37.21,36.65,40.60,50.75,37.14,38.04],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive Bayes</td>
      <td>0.78</td>
      <td>1.19</td>
      <td>1.20</td>
      <td>38.15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>0.62</td>
      <td>1.08</td>
      <td>1.04</td>
      <td>37.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>0.58</td>
      <td>1.05</td>
      <td>1.06</td>
      <td>37.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>0.56</td>
      <td>1.07</td>
      <td>1.09</td>
      <td>37.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Machines</td>
      <td>0.66</td>
      <td>1.10</td>
      <td>1.12</td>
      <td>36.65</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest [One Hot]</td>
      <td>0.83</td>
      <td>1.18</td>
      <td>1.18</td>
      <td>40.60</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest [Response Coding]</td>
      <td>0.07</td>
      <td>1.37</td>
      <td>1.40</td>
      <td>50.75</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR, SVM, NB] with LR</td>
      <td>0.80</td>
      <td>1.16</td>
      <td>1.14</td>
      <td>37.14</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Maximum Voting Classifier</td>
      <td>0.94</td>
      <td>1.21</td>
      <td>1.19</td>
      <td>38.04</td>
    </tr>
  </tbody>
</table>
</div>



# Appying Logistic Regression with Count Vectorizer (Unigrams & Bigrams)


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Logistic Regression [class balancing]','Logistic Regression [without class balancing]'],
    "Train (%)": [0.87, 0.86],
    "CV (%)": [1.19, 1.21],
    "Test (%)": [1.258, 1.254],
    "Misclassification (%)": [37.21, 37.03],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression [class balancing]</td>
      <td>0.87</td>
      <td>1.19</td>
      <td>1.258</td>
      <td>37.21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>0.86</td>
      <td>1.21</td>
      <td>1.254</td>
      <td>37.03</td>
    </tr>
  </tbody>
</table>
</div>



# Average Weighted Word2VEC Tabular Result


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Decision Trees','K' 'Nearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Classification', 'Support Vector Machines', 'Random Forest', 'Stacking [LR,SVC,NB] with K-NN', 'Stacking [Linear, K-NN, LR] with RF', 'Stacking [K-NN, SVC] with RF', 'Maximum Voting Classifier'],
    "Train (%)": [0.90,1.02,1.28,1.30,1.09,1.31,0.64,0.43,0.43,0.45,0.99],
    "CV (%)": [1.39,1.23,1.31,1.29,1.27,1.38,1.08,8.56,1.20,1.05,1.17],
    "Test (%)": [1.41,1.29,1.34,1.30,1.19,1.37,1.21,8.04,1.14,1.08,1.26],
    "Misclassification (%)": [43.04,42.85,48.68,45.11,46.05,0.5,37.59,38.34,41.50,37.74,45.41],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Trees</td>
      <td>0.90</td>
      <td>1.39</td>
      <td>1.41</td>
      <td>43.04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>1.02</td>
      <td>1.23</td>
      <td>1.29</td>
      <td>42.85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>1.28</td>
      <td>1.31</td>
      <td>1.34</td>
      <td>48.68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>1.30</td>
      <td>1.29</td>
      <td>1.30</td>
      <td>45.11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Classification</td>
      <td>1.09</td>
      <td>1.27</td>
      <td>1.19</td>
      <td>46.05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Support Vector Machines</td>
      <td>1.31</td>
      <td>1.38</td>
      <td>1.37</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest</td>
      <td>0.64</td>
      <td>1.08</td>
      <td>1.21</td>
      <td>37.59</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR,SVC,NB] with K-NN</td>
      <td>0.43</td>
      <td>8.56</td>
      <td>8.04</td>
      <td>38.34</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stacking [Linear, K-NN, LR] with RF</td>
      <td>0.43</td>
      <td>1.20</td>
      <td>1.14</td>
      <td>41.50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stacking [K-NN, SVC] with RF</td>
      <td>0.45</td>
      <td>1.05</td>
      <td>1.08</td>
      <td>37.74</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Maximum Voting Classifier</td>
      <td>0.99</td>
      <td>1.17</td>
      <td>1.26</td>
      <td>45.41</td>
    </tr>
  </tbody>
</table>
</div>



#  TF-IDF Average Weighted Word2VEC Tabular Result


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Decision Trees','K' 'Nearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Classification', 'Support Vector Machines', 'Random Forest', 'Stacking [LR,SVC,NB] with K-NN', 'Stacking [Linear, K-NN, LR] with RF', 'Stacking [K-NN, SVC] with RF', 'Maximum Voting Classifier'],
    "Train (%)": [1.83,1.83,1.83,1.83,1.83,1.83,1.83,8.67,1.83,1.83,1.83],
    "CV (%)": [1.83,1.83,1.83,1.83,1.83,1.83,1.83,8.66,1.83,1.83,1.83],
    "Test (%)": [1.83,1.83,1.83,1.83,1.83,1.83,1.83,8.69,1.83,1.83,1.83],
    "Misclassification (%)": [71.24,71.24,71.24,71.24,71.24,71.24,71.24,86.31,71.27,71.27,71.27],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Decision Trees</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Classification</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Support Vector Machines</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.24</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR,SVC,NB] with K-NN</td>
      <td>8.67</td>
      <td>8.66</td>
      <td>8.69</td>
      <td>86.31</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stacking [Linear, K-NN, LR] with RF</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stacking [K-NN, SVC] with RF</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.27</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Maximum Voting Classifier</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>1.83</td>
      <td>71.27</td>
    </tr>
  </tbody>
</table>
</div>



#  TF-IDF Top 2K Features with (n_gram = 1,4) Tabular Result


```python
import pandas as pd
dataframe = pd.DataFrame(
{
    "Maching Learning Model": ['Naive Bayes','KNearest Neighbours','Logistic Regression [class balancing]','Logistic Regression [without class balancing]', 'Support Vector Machines', 'Random Forest [One Hot]', 'Random Forest [Response Coding]', 'Stacking [LR, SVM, NB] with LR', 'Stacking [LR,Linear,RF] with LR', 'Stacking [NB, LR, RF] with LR', 'Stacking [K-NN, NB, Linear] with LR', 'Stacking [LR, K-NN, Linear] with LR', 'Stacking [LR, Linear] with LR', 'Maximum Voting Classifier'],
    "Train (%)": [0.58,0.63,0.45,0.45,0.59,0.87,0.05,0.38,0.46,0.43,0.44,0.42,0.43,0.88],
    "CV (%)": [1.232,1.060,1.026,1.056,1.072,1.191,1.293,1.083,1.057,1.089,1.090,1.060,1.072,1.067],
    "Test (%)": [1.177,1.041,0.943,0.962,0.996,1.135,1.246,1.15,1.112,1.158,1.169,1.131,1.135,1.039],
    "Misclassification (%)": [40.03,37.78,34.39,34.96,35.52,43.23,50.37,36.69,35.48,38.64,39.39,35.63,34.73,31.87],
}

)
dataframe
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Maching Learning Model</th>
      <th>Train (%)</th>
      <th>CV (%)</th>
      <th>Test (%)</th>
      <th>Misclassification (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive Bayes</td>
      <td>0.58</td>
      <td>1.232</td>
      <td>1.177</td>
      <td>40.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNearest Neighbours</td>
      <td>0.63</td>
      <td>1.060</td>
      <td>1.041</td>
      <td>37.78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression [class balancing]</td>
      <td>0.45</td>
      <td>1.026</td>
      <td>0.943</td>
      <td>34.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression [without class balancing]</td>
      <td>0.45</td>
      <td>1.056</td>
      <td>0.962</td>
      <td>34.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Machines</td>
      <td>0.59</td>
      <td>1.072</td>
      <td>0.996</td>
      <td>35.52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest [One Hot]</td>
      <td>0.87</td>
      <td>1.191</td>
      <td>1.135</td>
      <td>43.23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Random Forest [Response Coding]</td>
      <td>0.05</td>
      <td>1.293</td>
      <td>1.246</td>
      <td>50.37</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Stacking [LR, SVM, NB] with LR</td>
      <td>0.38</td>
      <td>1.083</td>
      <td>1.150</td>
      <td>36.69</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Stacking [LR,Linear,RF] with LR</td>
      <td>0.46</td>
      <td>1.057</td>
      <td>1.112</td>
      <td>35.48</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stacking [NB, LR, RF] with LR</td>
      <td>0.43</td>
      <td>1.089</td>
      <td>1.158</td>
      <td>38.64</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Stacking [K-NN, NB, Linear] with LR</td>
      <td>0.44</td>
      <td>1.090</td>
      <td>1.169</td>
      <td>39.39</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stacking [LR, K-NN, Linear] with LR</td>
      <td>0.42</td>
      <td>1.060</td>
      <td>1.131</td>
      <td>35.63</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Stacking [LR, Linear] with LR</td>
      <td>0.43</td>
      <td>1.072</td>
      <td>1.135</td>
      <td>34.73</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Maximum Voting Classifier</td>
      <td>0.88</td>
      <td>1.067</td>
      <td>1.039</td>
      <td>31.87</td>
    </tr>
  </tbody>
</table>
</div>



# 6. Observations

*As we can observe from the above tabular results, I've applied various techniques to reduce the test and cv erro below 1*

*I've applied various feature generation techniques like Bag of words, TF-IDF, Average Word2Vec, TF-IDF Weighted Word2Vec on cancer diagnosis dataset and printed the results in a tabular form*


1. **Bag of Words**:-

*Stacking Models have obtained better results when compare to other classifiers. Among them the best result we got by stacking Logistic Regression, Linear Regression and Random Forest and using Logistic Regression as the classifier*.


Train Error: 0.65,
CV Error   : 1.14,
Test Error : 1.08,
Misclassification Error : 35.33


*Apart from Stacking models, K-NN also obtained a good result*.

Train Error: 0.70,
CV Error   : 1.10,
Test Error : 1.09,
Misclassification Error : 39.66


2. **Term Frequency - Inverse Document Frequency (TF-IDF)**:-

*K Nearest Neighbours have obtained the best result*


Train Error: 0.61,
CV Error   : 1.05,
Test Error : 1.05,
Misclassification Error : 35.90


*The second best result is from Logistic Regression [with class balancing]*

Train Error: 0.57,
CV Error   : 1.15,
Test Error : 1.09,
Misclassification Error : 37.59


3. **TF-IDF with Top 1k Features**

*K Nearest Neighbours have obtained the best result*

Train Error: 0.62,
CV Error   : 1.08,
Test Error : 1.04,
Misclassification Error : 37.59


*The second best result is from Logistic Regression [with class balancing]*

Train Error: 0.58,
CV Error   : 1.05,
Test Error : 1.06,
Misclassification Error : 37.03


4. **Logistic Regression with Count Vectorizer (Unigrams & Bigrams)**

*We've applied logistic regression with count vectorizer including both unigrams and bigrams*.

*Logistic Regression without class balancing result is some what better when compare to logistic regression with class balancing result*

Train Error: 0.86,
CV Error   : 1.21,
Test Error : 1.25,
Misclassification Error : 37.03

5. **Average Weighted Word2VEC**


*Stacking K Nearest Neighbours & Support Vector Machines and Random Forest as Classifier, we obtained the best result compare to other classifiers*.

Train Error: 0.45,
CV Error   : 1.05,
Test Error : 1.08,
Misclassification Error : 37.74


6. **TF-IDF Top 2K Features with (n_gram = 1,4)**


*Among all the classifiers/models we've used, TF-IDF with top 2k features including ngrams (1,4) got the best result reducing the test error to below 1. Below are the top three best results*


> Logistic Regression with Class Balancing

Train Error: 0.45,
CV Error   : 1.02,
Test Error : 0.94,
Misclassification Error : 34.39


> Logistic Regression without class balancing


Train Error: 0.45,
CV Error   : 1.05,
Test Error : 0.96,
Misclassification Error : 34.96


> Support Vector Machines


Train Error: 0.59,
CV Error   : 1.07,
Test Error : 0.99,
Misclassification Error : 35.52

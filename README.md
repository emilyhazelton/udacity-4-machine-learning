# Project Summary

For my fourth Udacity project, I used machine learning to predict persons of interest in the Enron scandal. The data sets and definition for persons of interest were provided by the course instructor. 

I worked within the poi_id.py file to complete the following project tasks:

* Identify outliers in the financial data and explain how they are removed or otherwise handled
* Implement at least one new feature and test its impact on the final algorithm
* Attempt at least two different algorithms and compare their performance
* Deploy GridSearchCV for parameter tuning
* Use two metrics for evaluating algorithm performance
* Assess the final algorithm by splitting the data into training and test data
* Submit a final algorithm solution that, when tested, shows precision and recall as at least 0.3

Below, I include the project questions and write-up that I completed to document my process. 

_ _ _ 



## Project Questions 



*1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  
[relevant rubric items: “data exploration”, “outlier investigation”]*



The provided data set combines compensation information and email-patterns for people who worked at Enron. The instructor collected the list and went through it to identify persons of interest (POIs). POIs are people who met one of the following criteria during the Enron trials in 2001: indicted, settled without admitting guilt, testified in exchange for immunity from prosecution. 



The only outlier in the data was a "TOTAL" row that came from the financial dataset. I did not remove any of the individuals' information, as any outliers could indicate a pattern in being a POI. After removing the TOTAL row, there are 145 people on the list, 18 of which are identified as POIs. Also, of note, is that many of the financial fields are N/A. 



The goal of this project is to build an algorithm to predict if someone is a POI or not. 



_ _ _ 



*2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  
[relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]*



I used three features in my POI identifier -- salary, exercised_stock_options, and bonus. As part of my research, I watched The Smartest Guy in the Room, a documentary about the Enron case. Thus, I knew from the findings during the case that people in the inner circle who knew about the shady accounting practices of the company cashed out on their stock options before the company went bankrupt. I also figured that salaries and bonuses would be related to highly paid insiders at the company. I tried combining other financial indicators with these or removing some of these, but the blend of these three seemed to do best. I also tried adding in an email feature (from_poi_to_this_person), with the rational that people receiving a lot of emails from a POI may be a POI themselves. It actually made my classifier do worse, so I removed it. I did not need to do any scaling, as the GaussianNB and DecisionTree classifiers didn't require it. 



For a new feature, I created a feature named suspicious_email_weight. The value ranges between 0 and 1.  The value indicates what proportion of from and to messages that the person received/sent involve POIs. For missing values, I substituted -1 since DecisionTree does not accept nan values (this provided slightly better scores than simply substituting 0; this makes sense as the indicator can differentiate between a missing value and someone who simply has never emailed with a POI this way).  These scores came from my final algorithm combo, as described in the next question. 



* With missing values as -1: Accuracy: 0.84813       Precision: 0.42446      Recall: 0.39050

* With missing values as 0:  Accuracy: 0.82515       Precision: 0.42344      Recall: 0.37750


Unfortunately, I determined I shouldn't use suspicious_email_weight on the final calculation, as it slowed the process, and had mixed results in the scores. It increased precision by about a hundredth, but also reduced recall by the same amount. Accuracy did go up by .02, but my understanding is that accuracy on a dataset with such a skew in labels (18/145 as POIs) is not very meaningful. I did not find that the slowed processing time made it worth adding this feature. I saved a csv file named 'grid_results.csv' with the full matrix of tests. 



### Notes for submission #2



I implemented SelectKBest in a new file named 'poi_id_kbest.py. 



Using the best features, according to the selector, I tried the following options with PCA in the pipeline, with the scores from tester.py shown. 



1. DecisionTree with 'suspicious_email_weight', 'director_fees', 'total_payments', 'deferral_payments' -->  Precision: 0.20465      Recall: 0.20700

2. DecisionTree with 'director_fees', 'total_payments', 'deferral_payments', 'exercised_stock_options' -->  Precision: 0.28763      Recall: 0.26750

3. GaussianNB with 'director_fees', 'total_payments', 'deferral_payments', 'exercised_stock_options' --> Precision: 0.46435      Recall: 0.27350



Then, I tried removing PCA. 



1. A simple GaussianNB with 'director_fees', 'total_payments', 'deferral_payments', 'exercised_stock_options' --> Precision: 0.19067      Recall: 0.41300

2. DecisionTreeClassifier with min_samples_split=6 (which was the best tuning selected earlier in GridSearchCV in combo with PCA) 



At this final step, I finally got --> Precision: 0.32253      Recall: 0.31350



However, with all this work, I can't get KBest features to provide the performance that I got when I used my human knowledge of the case and selected what I thought would be the strongest features as a result. The code in poi_id.py gives me --> Precision: 0.46588      Recall: 0.41650



I'm not sure what I'm doing wrong with SelectKBest. Any feedback would be welcome! 



### Notes for submission #3



I moved SelectKBest into the GridSearchCV process. I used Feature Union to combine PCA and SelectKBest, so I don't have a list of which features were selected. I did not do any feature scaling in advance because DT does not require it. I provided GridSearchCV with 13 features -- all financial features that had values for at least 50% of the records and my created 'suspicious_email_weight' feature, which merges several of the email count features. In the best estimator, PCA chose 3 components and SelectKBest chose 4.  I compared SelectKBest using both the f_classif and mutual_info_classif, and mutual_info_classif had better predictive power. I chose these two parameter values because they were listed as good options for classifier algorithms that also handle negative numbers (which was necessary for some of the financial features; chi2 was not an option for this reason). In multiple tests that I rand with GridSearchCV, recall kept coming back very low. As a result, I adjusted GridSearchCV to score based on recall_micro to boost it above a recall score of .30 (I also tried it with scoring set to recall and recall_macro, and recall_micro gave me the bost desirable result, even better than f1).  


_ _ _ 



*3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  
[relevant rubric item: “pick an algorithm”]*



I tried GuassianNB, SVC, Nearest Neighbor, and DecisionTree Classifier. I quickly learned that Nearest Neighbors is for unsupervised learning and does not have a predict function. I was not able to get SVC working either. 



Between GuassianNB and DecisionTree, things were pretty close. With exercised_stock_options as the only feature and GaussianNB, my scores were as follows: 



* Accuracy: 0.90409       Precision: 0.46055      Recall: 0.32100 



When I used two features, salary and exercised_stock_options, I got: 


* Accuracy: 0.84946       Precision: 0.52192      Recall: 0.25600 



So the exercise is somewhat subjective in that different algorithms optimize specific scores. I opted to use an algorithm that balanced boosting both Precision and Recall. So I landed with a DecisionTreeClassifier. My scores were: 


* Accuracy: 0.82377       Precision: 0.42346      Recall: 0.40250



I was happy with both precision and recall scoring higher than 0.4, as the project requirement was to score higher than 0.3. 


_ _ _ 



*4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  
[relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]*



I deployed GridSearchCV to tune the parameters of my PCA and the algorithm that I used. I tuned the number of components for PCA and min_samples_split for DecisionTree. 



The overall goal of parameter tuning to set your algorithm up in a way that it performs in the best way as possible. Specific scoring metrics allow you define what "best" performance means. The parameters that give you a better result are "tuned" vs "untuned." A balance in the algorithm's capacity to generalize well to a new set of data is the key. You want to make sure that your algorithm provides ENOUGH variation to reflect the real data set (avoid underfitting) and that it does not provide TOO MUCH variation to reflect real data patterns (avoid overfitting). I found the [Underfitting vs. Overfitting](http://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) page on Scklearn to clearly explain this with a cosine graph.



_ _ _ 



*5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  
[relevant rubric items: “discuss validation”, “validation strategy”]*



Validation is the practice of reserving a portion of your data during the training process so you can test how well your algorithm generalizes when it sees new data. It serves as a check on overfitting. 



Validation can go wrong if your labels are sorted, and one label ends up in your test data and the other(s) end up in training data. Thus, shuffling the data as it is split between the train and test sets is a good idea. 



I used the StratififiedShuffleSplit from tester.py to evaluate my algorithm performance. I did investigate various methods of testing, though (e.g. also looked at how StratifiedKFold would be implemented.  



_ _ _ 



*6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. 
[relevant rubric item: “usage of evaluation metrics”]*



The precision of my algorithm was 0.42. This means that when the algorithm said someone was a POI, it was actually true 42 percent of the time. 



The Recall was .40. This means that of all the POIs that the algorithm saw, it correctly identified 40 percent of them as a POI (and missed 60 percent).



_ _ _



## Additional Notes  


I consulted the following resources during my work:



* Scikit learn documentation 

* Course examples and course files

* Stack overflow error messages and answers



I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.



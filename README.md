# MLProject-Customer-Churn-EDA-and-Prediction  

## Overview  
This repository hosts a Python project dedicated to predicting customer churn within the telecommunications sector. Leveraging a dataset of customer interactions and demographics, I embark on a journey through rigorous Exploratory Data Analysis (EDA) to uncover underlying patterns, behaviors, and factors contributing to customer churn. Building upon these insights, I then develop and refine a machine learning classifier model designed to accurately forecast potential churn, enabling targeted customer retention strategies.  

## Data
The [data](https://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-datasethttps://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-dataset) used in this project was gotten from Kaggle. There were five (5) csv files containing different information about the customers. 

## Tools
- Python: For data processing and modeling.  
- Pandas & NumPy: For data manipulation.   
- Matplotlib & Seaborn: For data visualization.  
- Random Forrest Classifier: For predictive modeling.    
- Scikit-learn: For model evaluation and metrics calculation.

## Process  
The EDA process started with merging the five datasets containing all the information I needed. Right before that, I had to do an initial cleanup of the column names to ensure they were consistent with the recommended snake_cased naming convention. After the data merge, I had a dataset of 7,043 rows and 56 columns. 

I checked for nulls and the data appeared to be mostly clean, except for two columns 'churn_reason' and 'churn_cat'; which both tell us the reason why customers say they churn (about 15 unique reasons) and the churn reason category respectively. To simplify things, the churn reason category was created to group the 15+ possible reasons into 5 distinct categories that capture all the possible churn reasons.   

Out of the 7043 observations, there were 5174 nulls in both these two columns. This is a very high number of null data so I could not just discard the entire column without due process. I sought to uncover the reason for this staggering amount of nulls and found that these represented the customers who did not churn. No data was recorded in these two columns hence the reason why it was assigned NaN. I proceeded to fill the null values with 'None' to represent the customers that did not churn.  

Once the nulls were taken care of, I proceeded to exploratory data analysis to answer the questions that naturally came up with the information in the dataset.  

## Exploratory Data Analysis  
I started my EDA by checking the percentage of customers that churned as opposed to those that did not. I found a highly imbalanced dataset of 73% not churned; 27% churned. This overall churn ratio served as my baseline for comparing the behavior of specific segments within this dataset.   

### How does Gender affect Churn?  

![Relationship beween Churn   Gender variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/f7b60cab-fbb7-4bf4-ae83-6f94b227e051)  

The churn ratio for customers who are male and female aligns closely with the overall churn ratio. This suggests that customers in both gender groups behave similarly to the average customer with respect to churn. Therefore, gender does not significantly affect churn behavior and might not be a strong predictor of churn on its own.

### How does being below or above 30 affect customer churn?  

![Relationship beween Churn   Under 30 variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/68961b84-f921-4f5c-8866-eaed968a3197  

For those under 30, 78% did not churn while 22% churned. For those over 30, 72% did not churn and 28% did. The churn ratios in both groups here (under and above 30) are similar to the overall churn ratio of 73% and 27% with only slight variations. We cannot explicitly say this variable affects churn on its own.  

### How does senior citizenry affect churn?  

![Relationship beween Churn   Senior Citizenry variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/b341b284-36d1-4698-b0ff-f2e38b75fe5b)  

Non-Senior Citizens (74% not churned, 26% churned): The churn ratio for customers who are not senior citizens aligns closely with the overall churn ratio. This similarity suggests that the non-senior citizen group behaves similarly to the average customer with respect to churn. Therefore, being a non-senior citizen, in this case, does not significantly deviate from the expected churn behavior and might not be a strong predictor of churn on its own.

Senior Citizens (58% not churned, 42% churned): The churn ratio for senior citizens, however, significantly differs from the overall churn ratio, with a much higher percentage of churn. This indicates that senior citizens as a group are more susceptible to churning compared to the average customer. The significant deviation from the overall churn rate suggests that senior citizen status might be an important factor or predictor of churn.  

#### Why do senior citizens churn?  
Now that we have established that senior citizens churn. I sought to understand why because in the list of churn reasons, 'Deceased' was among. I wanted to ensure this was not the leading cause of churn among senior citizens.  

![Why Senior Citizens Churn](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/b547b5ee-06e9-40de-b010-a4bbd12b107c)  

From the analysis, the majority of the reasons why senior citizens churn is because of competitors, customer support attitude, and price.  

### How does being married affect churn?  

![Relationship beween Churn   Married variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/4c5c46bf-f58f-488a-8bc9-f00e6efb3fa7)  

We see that the churn ratio for the married variable deviates significantly, in opposite directions, for both groups. Considering the overall churn ratio of 73% not churned and 27% churned, we see that those who are not married tend to churn more (67% not churned, 33% churned) than those who are married (80% churned, 20% not churned).  

### How does having dependents affect churn?  

![Relationship beween Churn   Dependents variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/5c9b0e2a-afa1-4891-894f-56db3e4a60f6)

From our analysis, we see that those who have dependents tend to even be more loyal as 93% of this group are retained while only 7% are churned. While those without dependents, with a ratio of 67% not churned; 33% churned, are slightly more susceptible to churning.  

### How does giving referrals affect churn?  

![Relationship beween Churn   Referred_a_friend variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/d4a125dc-dc3d-4288-bc8f-ba95ca915c05)

As expected, those who refer a friend tend to be more loyal than those who do not refer a friend (considering the churn ratio baseline of 73% to 27%). Those who referred a friend had a churn ratio of 81% not churned; 19% churned. While those who did not refer a friend had a churn ratio of 67% not churned; 33% churned.  

### How does having phone service affect churn?  

![Relationship beween Churn   Phone_service variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/bfc14028-cf6a-45b5-b6cb-775ed2478c24)

- A lot more people who have phone service use this company's telecom service.
- Both groups (i.e. customers that use phone service and those that do not) do not deviate from the churn ratio baseline (73% not churned; 27% churned) because those that use phone service have a churn ratio of 73% not churned; 27% churned while that do not have phone service so we cannot say this feature significantly affects churn in any meaningful way on its own.

### How does having multiple lines affect churn?  

![Relationship beween Churn   Multiple_lines variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/09d7653a-84a6-4fe7-b318-e49a6b0ecb5c)  

Customers with multiple lines => 71% not churned, 29% churned. Customers without multiple lines => 75% not churned, 25% churned. Having multiple lines or not does not significantly affect churn as the churn ratio of both groups does not deviate significantly from the baseline churn ratio.  

### How does having internet service affect churn?  

![Relationship beween Churn   Internet_service variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/ee68359a-8a1b-462c-b5a3-b0a5b88c704d)  

An interesting phenomenon develops here; those without internet service tend to be more loyal (93% not churned; 7% churned) than those with internet service (68% not churned; 32% churned).  

### How does having online security affect churn?  

![Relationship beween Churn   Online_security variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/bd9b6802-6414-46e2-8d28-ed9af135bd89)  

Customers with online security => 85% not churned, 15% churned. Customers without multiple lines => 69% not churned, 31% churned. Those without online security tend to be churners than those with online security, although the deviation from the baseline for the latter is not too wide. Those with online security, on the other hand, tend to be very loyal.  

### How does having unlimited data affect churn?  

![Relationship beween Churn   Unlimited_data variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/1a0815c7-f883-4b62-924a-e871f3f8dff8)  

Customers having unlimited data => 68% not churned, 32% churned. Customers without unlimited data => 84% not churned, 16% churned. Looking at the percentages of both groups we see that they both deviate from the baseline significantly. Interestingly, customers who have unlimited data tend to be more churners than customers who do not have unlimited data. This could correspond to the 'have phone service' feature because in that case, customers who had phone service were the churners over those who did not. And it is the customers that have phone service that have higher chances of having unlimited data (or better still, being on the unlimited data plan service).  

### How does having premium tech support affect churn?  

![Relationship beween Churn   Premium_tech_support variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/0b56adc8-a27b-4373-9562-3f4748ec0ebb)  

Customers with premium tech support => 85% not churned, 15% churned. Customers without multiple lines => 69% not churned, 31% churned. Deviation from the churn ratio baseline goes in the opposite direction for both groups. Customers without premium tech support tend to be churners while those with premium tech support are more loyal.  

### How does using paperless billing affect churn?  

![Relationship beween Churn   Paperless_billing variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/1c7fe597-cddc-487f-bb6a-24a36410f7a1)  

Customers that use paperless billing => 66% not churned, 34% churned. Customers that do not use paperless billing => 84% not churned, 16% churned. Customers who use paperless billing are more likely to be churners than those who do not use paperless billing.  

### How does payment method affect churn?  

![Relationship beween Churn   Payment_method variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/5568006f-7138-4a72-8939-aa50cf5d80f6)  

- Most of the customers of this telecom company use bank withdrawal payment method.
- Bank withdrawal customers (66% not churned;	34% churned).
- Credit card customers (86% not churned;	14% churned)
- Mailed check customers (63% not churned;	37% churned)

Of all the customer who use diverse payment methods, those who use their credit cards are more loyal than the rest. While those who use mailed checks are more likely to churn.  

### How does internet type affect churn?  

![Relationship beween Churn   Internet_type variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/ee555905-ee86-4be4-94da-bc739e7ac6e2)  

Most customers are fiber optic users, while the least number of customers use cable.
- Cable internet users (74% not churned;	26% churned)
- DSL internet users (81% not churned;	19% churned)
- Fibre optic internet users (59% not churned;	41% churned)
- Users who do not use any internet type (93% not churned;	7% churned)

41% of customers who use fiber optic internet are churners. This is several magnitudes higher than the churn ratio baseline. Customers who do not use any internet type are the least likely to churn. Those who use cable do not deviate significantly from the baseline and lastly, those who use DSL are also loyal and least likely to churn.  

### How does city affect churn?  

![Graph of top 5 cities where customers Churn and where they do not churn](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/b3d336c8-0caa-4780-89df-707d2bcc7531)  

Regarding the top 5 cities where customers churn, we have the following observations from the analysis:

- About 185 out of 285 customers churn in San Diego. That is a churn ratio of 35%:65% in favor of the churners. Way above the baseline chrun ratio of 73% not churned; 27% churned.
- In Los Angeles, 78 customers churn out of 293 total customers. which gives us a churn ratio of 73% not churned; 27% churned, which is the same as the overall baseline churn ratio.
- For San Francisco, we have 31 churners out of 104 customers, which amounts to 30% churned customers. This is slightly higher than our overall churner percentage but not a huge deviation.
- In San Jose, we have a churn ratio of 74% not churned and 26% churned.
- Lastly, 60% of the customers in Fallbrook are churners, out of 43 customers in total.

### How does offer affect churn?  

![Relationship beween Churn   Offer variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/2fd513c3-dee4-42b5-9c3b-564a6474beb3)

Most customers that were given an offer were given offers B and E.
- Customers given offer A (93% not churned;	7% churned)
- Customers given offer B (88% not churned;	12% churned)
- Customers given offer C (77% not churned;	23% churned)
- Customers given offer D (73% not churned;	27% churned)
- Customers given offer E (47% not churned;	53% churned)
- Customers given no offer (73% not churned;	27% churned)

The offer variable represents the last marketing offer that the customer accepted. Of all the categories in this group, customers who accepted Offer E are the most likely to churn. While those who accepted offers A and B are the most loyal and least likely to churn. The other groups are fairly consistent with the overall churn ratio of 73% not churned; 27% churned. Customers who are not given any offers produced the same churn ratio as the overall churn ratio. This means offers A and B can improve customer retention.  

### How does contract affect churn?  

![Relationship beween Churn   Contract variable](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/0d4b4489-de15-4e13-a749-f49c42baccb5)  

Over half of the customers of this telecom company are month-to-month subscribers.  
- Month-to-month customers (54% not churned; 46% churned)  
- One-year customers (89% not churned; 11% churned)  
- Two-year customers (97% not churned; 3% churned)

As the duration of the contract increases, the loyalty of the customers increases. Month-to-month subscribers are extremely likely to churn, while those on a two-year contract rarely ever churn.  

### Relationship between monthly charges and churn  

![Distribution of Monthly Charges](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/36bec15f-fb85-482a-b9c7-79d8c551482b)  

The distribution of the monthly_charge variable is multimodal with most customers paying $20 for this telecom service. There are other 'peaks' in this distribution albeit they are more broad. All these suggest the business could be offering different price tiers for their service.  

Although I suspected that this graph took into account the 'monthly' charges of customers on one-year and two-year contracts. So I might have a situation where there are indeed tiered monthly prices which are captured by this graph and also monthly prices of annual and bi-annual contracts which are also captured by this graph. I decided to investigate this further to be sure I interpret further analyses correctly.  

Upon Further analysis, I found that month-to-month customers pay as high as $117.45 per month and as low as $18.75. These values are similar to how much annual and bi-annual customers pay; as high as $118.75 and as low as $18.25 per month. This confirms my suspicion that there are indeed tiered month-month pricing plans that go all the up to $117 per month as well as annual and bi-annual plans that average monthly charges of up to $118. All of which will be captured by our monthly_charges distribution.  

With this understanding, we can now analyze monthly charges by churn;  

![Monthly charges by Churn](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/8f21aeca-46b2-4266-8875-dcc4d3253e7d)  

This graph tells us that as monthly charges increase, the churn also increases. But we need to interpret this graph carefully and with the context of our understanding of the churn behaviors of our three groups of customers and the monthly_charge distribution chart above.

From what we know about customers in this dataset, those on the one-year and two-year contracts are very loyal. However, from our analysis of their monthly charges, we see that they pay as high as $118 per month meaning these customers would be positioned at the rightward band of this chart where churn seems to be occurring.

Nonetheless, we also see from customer churn behavior that month-to-month customers are extremely likely to churn, and they also pay as high as $117 per month. Meaning they would also be positioned at the area of this graph where churn occurs.

With these contexts, I infer that as the monthly charges of Month-to-Month customers increase, so does their likelihood of churn.  

### Relationship between tenure_in_months and churn  

![Customer Tenure in Months by Churn](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/66d8eb61-0e03-48d2-b446-17e145c21aee)  

This graph tells us that customers tend to churn in the early months of their tenure. I decided to analyze further to find specific range of months that customers churn as this will be more useful to stakeholders.  

![Customer Tenure in Months Range by Churn](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/2d8c8c81-c72a-4c68-8a12-9a6a1b3daf96)  

From this chart, we see that most customers churn between the first 3 months of their tenure. This churn decreases as the customer's tenure increases.  

We should be careful to interpret the 36+ tenure months bar in the graph with caution because it represents churn numbers for a period of 3 years (year 2 - year 5). So it is only natural to see such a high number given the vast amount of time captured by that bar. The KDE plot above helps us put this bar graph into more perspective.  

### Why customers churn according to the reasons they give  

![Share of Churn Reason Among Customers](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/19ec2250-647a-4221-9a79-db9e8169b97d)  

This chart gives an overview of why the telecom company is losing customers. According to this, 45% of the reason customers are churning is because of competitors. Given the fact that an additional column was provided in this dataset that depicts the specificity of the reasons, I decided to analyze the data further to get more granular.  

![Share of Churn Reason Among Customers (Granular)](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/e657f753-1371-484c-b254-a1052900fa5f)  

The majority of the time, customers are churning because competitors have better devices and make better offers. Another thing we should be looking at is the customer support system. There seems to be an issue here because aside from the competition, bad customer support represents a big chunk of the reason why customers are churning.  


## Modelling  
Given the sheer number of columns in this dataset (49), I decided to start the modeling process by performing feature selection via correlation analysis. Here, I sought to remove and keep columns according to how well they can predict the target variable 'churn'.  

At this point, most will default to using the Pearson correlation coefficient. However, it is most effective when examining the relationship between continuous dependent and independent variables. Recognizing this limitation, I opted for alternative correlation coefficients that better match the nature of my data. For analyzing the relationship between a continuous dependent variable and a categorical target variable, I employed the Point-Biserial Correlation. For ordinal independent variables, Spearman's Rank Correlation was more appropriate. Additionally, Cramér's V was selected for its suitability with categorical independent variables.   

![correlation_dataframe](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/09dd7ef8-853b-4bb7-8e68-c41e3a56d423)  

After my correlation analysis, I removed features that had a relatively low correlation to the target variable 'churn'.  


Next, I prepared my data for modeling  by Label Encoding Binary Categorical Columns, One Hot Encoding Categorical Variables With More Than Two Categories, and Target Encoding the 'city' Categorical Variable With High Cardinality (over 1,000 unique instances).   

After this, I used the random forest classifier and got a model score of **0.9659332860184529**.  

### Model Evaluation  
With this high model score, there was little to no need to do hyperparameter tuning. So I proceeded to evaluate the model to see how it was performing. First, I checked the model's performance in training and test data and found that:  

Training set accuracy =  0.9884629037983671  
Test set accuracy = 0.9659332860184529  

The training and test accuracies are both high and relatively close which suggests that the model is generalizing well. However, the very high score for the training data accuracy suggests the model could be learning the training data perfectly and is capturing all its nuances, including noise. I decided to use cross-validation to ensure that the model's performance is stable across different subsets of the training data.  

Cross Validation (CV) Scores: [0.9524485450674237, 0.9460610361958836, 0.9581263307310149, 0.9580965909090909, 0.9588068181818182]  
Mean CV Score: 0.9547078642170461  

The consistent and high cross-validation scores across all folds tells us the model has low variance in its performance across different subsets of the data and it indicates that the model is stable and not highly sensitive to the specific data it is trained on.  

A mean CV score of approximately 0.955 is excellent and suggests that the model should perform similarly on unseen data.

![ROC curve](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/1b92b991-2d36-49b7-b76f-ed641ab41c3a)  

Plotting the ROC-AUC curve, we see an excellent model performance with an AUC of 0.995. An AUC of 0.995 suggests that there is a 99.5% chance that the model will be able to distinguish between a random positive and a random negative instance.  

![Classification Report](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/f9ab4306-c68c-408e-8a39-3f5d79f602f4)  

From the classification report, we see high precision, recall, and f1-scores indicating that the model is classifying the 'Churn' and 'No Churn' categories effectively.  

![Confusion Matrix](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/688653ef-2f46-44b4-9602-b39cba93328d)  

The confusion Matrix tells us the following  
- TN: The matrix shows that over 1,000 cases where the number of actual "No Churn" cases were correctly predicted by the model as "No Churn."  
- TP: The matrix shows 330 cases where the number of actual "Churn" cases are correctly predicted as "Churn." here, which suggests the model is reasonably effective at identifying customers who will churn.  
- FP: There are 22 cases where the model incorrectly predicted "Churn" when they were actually "No Churn." here, which is relatively low, indicating good precision.  
- FN: There are 52 cases where the model incorrectly predicted "No Churn" when they were actually "Churn."   
  
All these go to show that the model is great at predicting the churn of customers for this telecommunication dataset.

## Marketing Recommendations  
Here are my recommendations borne from the insights obtained from the exploratory data analysis:  

**Senior Citizen Churn:** The data tells us that senior citizens are more likely to churn, and it's not because they are deceased. They churn because competitors make better offers or provide better devices. I implore this company to consider implementing a 'Senior citizens' plan that will rival competitors. This campaign will not only target senior citizens who are likely to churn, but will incentivize other users to keep using the company's services till their old age.  

**Competitors:** Speaking of competitors, 45% of the reason customers churn us is because of competitors. This company needs to go back to the drawing board to offer appealing offers that will reduce that percentage. A good place to start will be improving the internet services. Customers who use the internet service are most likely to churn. Additionally, those who use the fibre optic cables of this comapny churned the most (41%), suggesting customer dissatisfaction with this product. 

**Customer Support:** An additional 17% of customers churn because of bad customer support. The telecom company needs to invest in quality customer support personnel and systems to ensure better service delivery.  

**Offers:** A whopping 53% of customers given offer E churned. Whatever offer E is, it's not working. The company should either refine the offer or dump it altogether and focus on offers A and B, which prove to give better customer retention.  


​





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







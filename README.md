# FightClub

The safety and well-being of students in educational institutions has always been a concern for faculty. A survey conducted through the years 2009 to 2019 by the National Center for Education Statistics reported that 18% of students have been in a school fight, and this has become a problem that school districts are attempting to solve.

In this project, we are continuing the work of using an AI image detection model to identify whether a fight is occurring between students on school property and automatically notify school faculty. While this technology cannot preemptively prevent fights, our goal is to use this technology to minimize risks and injuries by allowing faculty to have a faster response to brawls among students and detect these altercations earlier.

## How to run efficientnet
1. download dataset from kaggle at https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
2. Use `ExtractData.py` to unzip and separate values from dataset into test, train, and validate
3. Run `basecode.py` for the base version of efficientnet
4. Run `best_efficientnet.ipynb` for the version we found have the highest accuracy 

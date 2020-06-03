Yahoo! Answers Topic Classification Dataset

Version 2, Updated 09/09/2015


ORIGIN

The original Yahoo! Answers corpus can be obtained through the Yahoo! Research Alliance Webscope program. The dataset is to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with Yahoo!. The dataset is the Yahoo! Answers corpus as of 10/25/2007. It includes all the questions and their corresponding answers. The corpus contains 4483032 questions and their answers. 

The Yahoo! Answers topic classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).


DESCRIPTION

The Yahoo! Answers topic classification dataset is constructed using 10 largest main categories. Each class contains 140,000 training samples and 6,000 testing samples. Therefore, the total number of training samples is 1,400,000 and testing samples 60,000 in this dataset. From all the answers and other meta-information, we only used the best answer content and the main category information.

The file classes.txt contains a list of classes corresponding to each label. 

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 4 columns in them, corresponding to class index (1 to 10), question title, question content and best answer. The text fields are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".


Download Link:
You can download the original dataset here: https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset

The *pre-processed dataset in our paper* can be found here: https://drive.google.com/file/d/1IoX9dp_RUHwIVA2_kJgHCWBOLHsV9V7A/view?usp=sharing

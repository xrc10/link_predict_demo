# link_prediction
A regression based bilingual dictionary prediction method. The task is extending an existing (small-sized) bilingual dictionary via cross-lingual regression over embedded words. The inputs of the system are a small bilingual dictionary between source language and target language and the word embedding vector representations for words in both languages. The output of the system is an extended dictionary for our next step cross-lingual text classification.

The scripts runs the dictionary extention method. 
- transLearnMatInv.m : the function to learn the model parameter
- transEval2.m : script to evaluate the learned model
- demoRun.m : the example code for running of training, evaluating model and saving intermediate results(extended dictionary) for cross-lingual text classification.

All data lies in the data/ folder.
- en.svm : the vector representation for words in source language, line format is `<word_idx>` `<value_on_dimension_1>` `<value_on_dimension_2>`,...,`<value_on_dimension_d>`.
- `<target_language>`.norm.svm: the vector representation for words in target language, same format as above.
- dict.`<target_language>`.trn.txt: the dictionary to train, line format is `<source_word_index>` `<target_word_index>` `<label>`, where `<label>` = 1 if two words have similar meanings and `<label>` = 0 otherwise.
- dict.`<target_language>`.val.txt: the dictionary to validate the model, same format as above.

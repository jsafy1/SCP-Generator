# SCP-Generator

This program uses a Long Short Term Memory network to generate text in the style of [SCP Foundation articles] (http://scp-wiki.wikidot.com/). 

For the uninitiated, the SCP Foundation is a fictional organization which protects the world from supernatural threats. The reason I chose this to recreate is because:
1. Most articles are highly structured. There is an item number, object class (how dangerous it is), containment procedures, including things like dimensions of containment facilities, and a description. I wanted to see if the LSTM could replicate this format.
2. Some articles have redacted / censored parts. Articles handle these in two ways, either by using black bars, e.g. "Mr. ██████ ..." or by putting a [DATA EXPUNGED] message. These pose a challenge to the LSTM, similar to how [Google BERT] (github.com/google-research/bert) used masked input in order to train their model. I wanted to see if the network would learn how to censor effectively or if it would try to fill in the censored parts.
3. It was fun and no one has done this before.

The model in this repo has been trained for 6 hours on an Nvidia 970M, a.k.a. not a lot at all. The loss was decreasing every epoch so there is still a lot of room for improvement, and you can continue to train the model. 

This program accepts command line arguments. Type "python lstm.py -h" for a list of arguments and description of each. You can train a new model, continue training an existing model, or test a model to see what it outputs. 

Some example output:
NOTE: All input and output should be lowercase.

>the subject will attempt to speak and exhibit any recording of the containment cell in scp-1009 to scp 1112. scp-1020 is a static security clearance of the previous personnel on ██/██/19██ and the subject is able to appear to have shown the windows of a supervisor of the first thing and extend the subject in the subject's environment. 

This example is interesting because it shows that the model learned how to censor appropriately. The date censoring makes sense syntactically, as well as within the timeline of the SCP Foundation.

> It has been used in a safety test for the new model ███████ in the restraint of the subject and each of the specimens were then passed onto the containment chamber by dr. █████████████

This example also demonstrates an understanding of censorship. The model is able to censor both the doctor's name, which is common in the training data, but also "model ███████," which is interesting since the LSTM had to understand that it might make sense to censor something like a model number or specification. 

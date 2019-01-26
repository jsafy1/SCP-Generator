# SCP-Generator

This program uses a Long Short Term Memory network to generate text in the style of [SCP Foundation articles](http://scp-wiki.wikidot.com/). It comes with a pretrained model, training data (~5,000 articles, 30M characters), and a scraper to show how the data was obtained. Type "python lstm.py -h" for a list of command line arguments and description of each. You can train a new model, continue training an existing model, or test a model to see what it outputs. 

### For the uninitiated
The SCP Foundation is a fictional organization which protects the world from supernatural threats. Each article in the training data represents a different supernatural entity. These entities all exist within the same fictional universe and often reference one another, as well as different aspects of SCP lore. I chose this body of work because:
1. Most articles are highly structured. There is an item number, object class (how dangerous it is), special containment procedures, which includes the dimensions and building materials of containment facilities, and a description of the object. I wanted to see if the LSTM could replicate this format.
2. Some articles have redacted / censored parts. Articles handle these in two ways, either by using black bars, e.g. "Mr. ██████ ..." or by putting a [DATA EXPUNGED] message. These pose a challenge to the LSTM, similar to how [Google BERT](github.com/google-research/bert) used masked input in order to train their model. I wanted to see if the network would learn how to censor effectively or if it would try to fill in the censored parts.
3. There are a lot of references to other entities in these articles, and I wanted to see if the LSTM would learn how to sensibly reference things outside of the scope of the current article.
3. It was fun, a great learning experience, and no one has done this before.

The model in this repo has been trained for 6 hours on an Nvidia GeForce 970M, a.k.a. not a lot at all. The loss was decreasing every epoch so there is still a lot of room for improvement, so you can continue to train the model. 

### Some example output:

Note: All input and output should be lowercase. This lowers the burden on the model, since it reduces the classification problem by 26 outcomes. 

> it has been used in a safety test for the new model ███████ in the restraint of the subject and each of the specimens were then passed onto the containment chamber by dr. █████████████

This example demonstrates an understanding of censorship. The model is able to censor both the doctor's name, which is common to see in the training data, but also "model ███████," which is interesting since the LSTM had to understand that it might make sense to censor something like a model number or specification. 

>the subject will attempt to speak and exhibit any recording of the containment cell in scp-1009 to scp 1112. scp-1020 is a static security clearance of the previous personnel on ██/██/19██ and the subject is able to appear to have shown the windows of a supervisor of the first thing and extend the subject in the subject's environment. 

This example also shows that the model learned how to censor appropriately. The date censoring makes sense syntactically, as well as within the timeline of the SCP Foundation. It is also interesting because of the references made to other entities, or "SCPs." In the training data it would be rare to see so many references so close to one another, although it still makes sense syntactically. The LSTM seems to have difficulty understanding which entity it is referring to at any one time however. It will sometimes start an article referring to SCP-XXX and then refer to the same entity as SCP-YYY. 
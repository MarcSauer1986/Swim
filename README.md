# Virtual Swim Coach

## About
Ziel dieses Projektes ist es, ein Produkt zu entwerfen, welches die Schwimmtechnik automatisiert evaluieren kann. 
Somit können Technikschwächen aufgedeckt werden und in einem weiteren Schritt korrigierende Übungsanweisungen gegeben 
werden.

## Motivation
Die Leistung beim Schwimmen ist im Vergleich zu anderen Fortbewegungsformen (z.B. Laufen) stark technikabhängig. 
Dies lässt sich anhand der verschiedenen Medien (Wasser vs. Luft) erklären, in denen sich ein Körper fortbewegt. 
Da hierbei Wasser eine höhere Dichte als Luft besitzt, hat die Schwimmtechnik einen höheren Stellenwert im Hinblick auf 
die Leistung als die Lauftechnik dies beim Laufen hat. Die Schwimmleistung ist im Allgemeinen vom Vortrieb und aktivem 
Widerstand abhängig.
Aus den dargelegten Gründen sind Schwimmer stets um eine saubere (und somit schnelleren) Schwimmtechnik bemüht. 
Die Schwimmtechnik wird normalerweise von ausgebildeten Schwimmtrainern evaluiert. Dies erfolgt auf Grundlage der 
Erfahrung des Trainers. In der Regel ist dies mit erheblichen Kosten und Aufwand verbunden.

## Road map

- Motion caption sensor installation
- Data collection (2 classes: high elbow catch and deep catch)
- Data cleaning and formatting
- Exploratory data analysis
- Feature engineering and selection
- Establish a baseline (logistic regression with 1 feature) and compare several machine learning models (logistic 
regression with multiple features, decision tree, SVM, neural network)
on a performance metric
- Perform hyperparameter tuning on the best model to optimize it for the problem
- Evaluate the best model on the testing set
- Interpret the model results to the extent possible
- Draw conclusions and write a well-documented report

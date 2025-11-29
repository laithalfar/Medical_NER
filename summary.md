What is a medical NER? Medical Named Entity Recognition (medical NER) is a specialized subtask of Natural Language Processing (NLP) that involves automatically identifying and categorizing key medical and clinical information (entities) from unstructured text, such as electronic health records (EHRs), clinical notes, and scientific literature. 


For our medical health records, we want to automatically detect and reference medical terms within any given text. This tool will serve as a component for different applications, including clinical data analysis, electronic health record (EHR) summarization, and intelligent medical search engines.

A system capable of identifying mentions of diseases, symptoms, medications, and other relevant medical entities in unstructured text. Furthermore, it will link these detected terms to standardized medical ontologies to ensure consistency and facilitate further analysis.


## Pre-built libraries ##

First compare different libraries for medical NER which would save us quite a lot of work in building one from scratch to identify medical terms.

1.	medspaCy: Think of it as spaCy wearing a doctor’s coat.

Use if:
-	You want control
-	You want a clean pipeline integrated with spaCy
-	You will build your own model and linker



2. GLiNER: This is the shiny new transformer-based universal NER model.

Use if:
-	You want speed + flexibility
-	You want to avoid heavy dataset preparation
-	You are okay with adding your own ontology mapping layer

GLiNER is fast to experiment with but not inherently tied to medical ontologies.

3.	Bio-Epidemiology-NER: A pretrained biomedical NER package covering multiple entity types.

Use if:
- You want a pretrained medical NER with broad coverage
- You plan to write your own linking layer


## Why not train one from scratch? ##

Now let’s look at the possibility of building a custom NER library from scratch. Training a custom NER model from scratch using a naive neural network only works well when we have massive amounts of data to generalize from. But when we have limited data in a particular domain, training from scratch is not effective. Instead, using a pre-trained model and fine-tuning it for a few additional epochs is the way to go. 

Be careful, when training in libraries such as spaCy. The results should be fine for the new domain but what about the pretrained entities? They have completely vanished. This is fine if we don’t need the pre-trained knowledge and are more focused on the new domain data. And what if the pretrained entities are also necessary? As a side effect of finetuning we face “Catastrophic Forgetting”.

Catastrophic Forgetting is a phenomenon in artificial neural networks where the network abruptly and drastically forgets previously learned information upon learning new information. This issue arises because neural networks store knowledge in a distributed manner across their weights. When a network is trained on a new task, the optimization process adjusts these weights to minimize the error tightly for the new task, often disrupting the representations that were learned for earlier tasks.

Some of the implications are,
•	Models that require frequent updates or real-time learning, such as those in robotics or autonomous systems, risk gradually forgetting previously learned knowledge.
•	Retraining a model on an ever-growing dataset is computationally demanding and often impractical, particularly for large-scale data.
•	In edge AI environments, where models must adapt to evolving local patterns, catastrophic forgetting can disrupt long-term performance.

Given that there are NERs that have similar scopes to what we are looking to detect there is no need to go the extra mile and train one from scratch. Especially that we would have to get the training data ourselves from scratch which is a massive amount of work.

## Ontologies ##

Medical ontologies are structured vocabularies that represent medical and biomedical knowledge in a machine-readable format, using formal relationships to define concepts like diseases, symptoms, and treatments. They are used in healthcare and research to standardize data, improve the accuracy of information systems, facilitate data analysis, and enable consistent communication across different systems and users. 

SNOMED CT
A gigantic, hyper-detailed clinical terminology.
Useful for:
•	Clinical procedures

•	Diagnoses and findings

•	Body structures

•	Lab results

•	Clinical workflows

MeSH

Simpler, smaller, and curated for biomedical literature, not hospitals.

Useful for:
•	Research publications

•	Diseases

•	Chemicals

•	Anatomy

•	Epidemiology terms

MeSH is generally the best first ontology for students building their first training pipeline.


UMLS

A monster library that merges many ontologies together (MeSH, SNOMED, LOINC, ICD, RxNorm, etc.)


Useful for:
•	Universal linking

•	Synonym expansion

•	Cross-walking between ontologies

•	Normalizing very messy medical text

UMLS is best after you already understand basic NER and ontology mapping.

So given this information MeSH is the best ontology for our case of building our first training pipeline. This leaves us with a combination of medspaCy and MeSH for our set project.

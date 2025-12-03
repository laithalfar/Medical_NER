First results indicate very poor performance with low precision and recall across all entity types.

The model shows zero true positives for most entity types, indicating it's not detecting any of the expected entities correctly. This suggests the model needs significant improvement or retraining with better data. Mostly showing False Negatives only.

# Initial Evaluation Summary

- Dataset: MACCROBAT2018 (first 5 documents)
- Pipeline flags: `--entity-type DISEASE --entity-type SYMPTOM --entity-type MEDICATION`
- Result snapshot: **Precision = 0.000**, **Recall = 0.000**, **F1 = 0.000**
- Observations: 0 true positives across all labels; the evaluator reports only false negatives (missed gold spans) and a small handful of false positives (e.g., generic "pain" mentions) produced by the demo MeSH gazetteer.

# Diagnosis

1. **Label mismatch** – MACCROBAT uses categories such as `DISEASE_DISORDER`, `SIGN_SYMPTOM`, etc., whereas the current pipeline emits `DISEASE`, `SYMPTOM`, `MEDICATION`. Without mapping these label spaces, every gold annotation is counted as a miss.
2. **Tiny knowledge base** – only a few hard-coded MeSH entries are available; most domain concepts in MACCROBAT never appear in the gazetteer, so medspaCy cannot produce matching spans.
3. **Rule-only detection** – no statistical NER model has been trained/fine-tuned on MACCROBAT, so coverage is limited to literal matches of the demo aliases.

# Next Steps

1. **Load the full MeSH ontology** via `--mesh-xml desc2024.xml` (and optional MeSH→UMLS map) to expand coverage, then re-run `evaluate.py` so the pipeline can emit real medical concepts.
2. **Introduce label alignment** by mapping MACCROBAT labels (e.g., `DISEASE_DISORDER`) to internal `entity_type` strings before scoring, or by post-processing predictions to the dataset schema.
3. **Augment detection**:
   - Seed the knowledge base with MACCROBAT-specific gazetteers derived from the gold `.ann` files.
   - Add medspaCy `TargetRule`s for high-frequency entities missing in MeSH.
   - Optionally fine-tune a transformer-based NER model (e.g., GLiNER, spaCy `en_core_sci_lg`) on MACCROBAT and use medspaCy only for linking.
4. **Iterative evaluation** – after each enrichment, rerun `evaluate.py --output-json results.json` to track precision/recall/F1 and inspect the stored FP/FN samples to guide further refinement.

Updated results:
Documents evaluated: 200
Gold entities: 25092
Predicted entities: 20318
Precision: 0.450 | Recall: 0.364 | F1: 0.403

Per-label metrics:
  ACTIVITY           P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=108)
  ADMINISTRATION     P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=176)
  AGE                P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=206)
  ANATOMY            P=0.000 R=0.000 F1=0.000 (TP=0 FP=1622 FN=0)
  AREA               P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=43)
  BIOLOGICAL_ATTRIBUTE P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=10)
  BIOLOGICAL_STRUCTURE P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=2942)
  CHEMICAL           P=0.000 R=0.000 F1=0.000 (TP=0 FP=268 FN=0)
  CLINICAL_EVENT     P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=626)
  COLOR              P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=52)
  COREFERENCE        P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=313)
  DATE               P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=732)
  DETAILED_DESCRIPTION P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=2904)
  DIAGNOSTIC_PROCEDURE P=0.636 R=0.809 F1=0.712 (TP=3702 FP=2121 FN=876)
  DISEASE            P=0.408 R=0.862 F1=0.553 (TP=1177 FP=1711 FN=189)
  DISTANCE           P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=122)
  DOSAGE             P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=365)
  DURATION           P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=280)
  ENTITY             P=0.000 R=0.000 F1=0.000 (TP=0 FP=2210 FN=0)
  FAMILY_HISTORY     P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=81)
  FREQUENCY          P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=76)
  HEIGHT             P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=4)
  HISTORY            P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=393)
  LAB_VALUE          P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=2864)
  MASS               P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=2)
  MEDICATION         P=0.701 R=0.900 F1=0.788 (TP=968 FP=412 FN=108)
  NONBIOLOGICAL_LOCATION P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=354)
  OCCUPATION         P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=13)
  OTHER_ENTITY       P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=21)
  OTHER_EVENT        P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=27)
  OUTCOME            P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=42)
  PERSONAL_BACKGROUND P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=57)
  PROCEDURE          P=0.000 R=0.000 F1=0.000 (TP=0 FP=527 FN=0)
  QUALITATIVE_CONCEPT P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=41)
  QUANTITATIVE_CONCEPT P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=31)
  SEVERITY           P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=369)
  SEX                P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=191)
  SHAPE              P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=66)
  SUBJECT            P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=54)
  SYMPTOM            P=0.653 R=0.717 F1=0.684 (TP=2411 FP=1280 FN=950)
  TEXTURE            P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=46)
  THERAPEUTIC_PROCEDURE P=0.464 R=0.882 F1=0.608 (TP=886 FP=1023 FN=119)
  TIME               P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=58)
  VOLUME             P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=33)
  WEIGHT             P=0.000 R=0.000 F1=0.000 (TP=0 FP=0 FN=4)

Sample false positives (up to 5):
  • [ENTITY] 'history' (209-216)
  • [ENTITY] 'time' (669-673)
  • [THERAPEUTIC_PROCEDURE] 'embolization' (3274-3286)
  • [DISEASE] 'metastasis' (72-82)
  • [DIAGNOSTIC_PROCEDURE] 'findings' (2224-2232)

Sample false negatives (up to 5):
  • [SYMPTOM] 'hypertrophy' (1885-1896)
  • [DIAGNOSTIC_PROCEDURE] 'echocardiography' (1011-1027)
  • [LAB_VALUE] 'maximum temperature of 48 °C' (2891-2919)
  • [SYMPTOM] 'lesion' (2351-2357)
  • [DETAILED_DESCRIPTION] 'exceeding the IC50' (3333-3351)

Detailed metrics saved to results_megakb.json
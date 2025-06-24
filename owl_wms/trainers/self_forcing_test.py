#### TESTING SAMPLERS - Goal:
## I suspect our sampling is broken because generation is completely nonsensical even though the causal
## student is initialized with bidirectional teacher's weights.
## 1. A) Load bidirectional model.
## 1. B) Load causal model as causal.
## 2. For each model:
## 2. A) Test Sampler as in av_trainer by saving .png's.
## 2. B) Test RF Sampler as in GameRFT by saving .png's.
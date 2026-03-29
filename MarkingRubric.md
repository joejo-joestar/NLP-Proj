# CS F429 NLP Midsem — Marking Rubric (Track B) [Quick Notes]

Source: Midsem Evaluation Rubric PDF.  [oai_citation:0‡Midsem_Assignment_NLP_CS_F429.pdf](sediment://file_00000000d630720881444ffc1e49e00b)

## (don’t lose marks)
- Use **held-out TEST set** for every table cell (real numbers only; no placeholders; no validation numbers).  
- Treat test set as **unseen** until evaluation (no re-running after seeing test outcomes).  
- **Live demo** will be run on an **unseen evaluator input**, and **any member** can be asked to explain any code section.  

## Marks summary (Track B total = 15)
1) **E1 & E2: Composite AUROC on RAGTruth + layer localisation** — **4 marks**  
   - Run all representation metrics, build composite, and include **layer-profile plot**.  
   - **μ_l and Σ_l estimated on TRAIN split only** (leakage check).  
   - 4/4 if **Composite AUROC ≥ 0.70 (test)** + layer plot + beat both baselines.  

2) **E3: Causal intervention (activation patching)** — **2 marks**  
   - Patch hidden states **both directions**: faithful→hallucinated AND hallucinated→faithful. 
   - **≥ 50 patching experiments**, report **CIE** by component type + significance (p<0.05) in ≥2 components + compare to ReDeEP.  

3) **E4: Temporal precedence (t−3 … t+1)** — **2 marks**  
   - Report drift at positions t−3 to t+1 + **line plot**.  
   - 2/2 if any metric peaks at **t−2 or earlier** + **Mann–Whitney U test** included.  

4) **E5: Cross-domain transfer to HaluEval (zero-shot)** — **2 marks**  
   - Run composite on HaluEval **without refitting μ_l, Σ_l, or PCA**.  
   - 2/2 if **Composite AUROC ≥ 0.62** + identify most brittle metric and explain why. 

5) **E6–E8: Deeper analysis** — **2 marks**  
   - E6: FFN vs attention drift decomposition  
   - E7: ≥ 3 failure cases with mechanistic explanations  
   - E8: SOTA gap table + structured gap analysis (upper bounds: ReDeEP ~0.82, LUMINA ~0.87).  
   - Full marks if all three done + close ≥50% of SOTA gap over baseline.  

6) **Code & Demo (live pipeline)** — **3 marks**  
   - Must run end-to-end on evaluator’s unseen input: hidden-state extraction + compute all 5 rep metrics + token-level outputs.  
   - Evaluator may ask: “show where Mahalanobis is computed; explain μ_l and Σ_l” (without warning).  

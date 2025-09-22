An automatic image restoration tool based on AgenticIR (https://github.com/Kaiwen-Zhu/AgenticIR) and CLIP4CIR (https://github.com/ABaldrati/CLIP4Cir/tree/master). 

We use CLIP4CIR to extract embeddings and degradation sequences of low-quality images, then store them to PostgreSQL for retrieval when new low-quality images come. Therefore, the processing time and cost will be shortened considerably.

Pipeline:  
1. Dockerfile can generate environment for the models called in AgenticIR and CLIP4CIR. You can use "conda env list" to check them.
2. Run "sh synthesize.sh" to generate synthesized low-quality data.
3. Run "python -m pipeline.infer" to generate restoration results and saved in output. One can first use evaluate_degradation_by="depictqa" to generate initial outputs. After saving enough knowledge in Step 4, one can use evaluate_degradation_by="clip_retrieval" for efficienty.
4. Refer to AgenticIR/retrieval_database/CLIP4CIR/run_pipeline.sh to train the model for image quality classification and insert history knowledge to PostgreSQL


![image](https://github.com/blackbean001/Auto-Image-Restoration/blob/main/pngs/pipeline.png)


![image](https://github.com/blackbean001/Auto-Image-Restoration/blob/main/pngs/result1.png)
![image](https://github.com/blackbean001/Auto-Image-Restoration/blob/main/pngs/result2.png)

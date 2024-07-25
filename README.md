# MistralFT-Gutenberg
Chosen LLM: mistralai/Mistral-7B-v0.1 

## Data preparation
To fine-tune LLM few books has been chosen from https://gutenberg.org/:
* A notched gun by Walt Coburn
* Arthur
* Electro-episoded in A.D. 2025 by E. D. Skinner
* Farmer Bluff's dog Blazer
* Rivals of the clouds by Raoul Whitfield
* The Cornhill Magazine (Vol. XLI, No. 241 new series, July 1916) by Various
* The invisible master

These books has been converted into 4 different inputs (asking to make a summary) and outputs (summaries). Three sets has been left for training, one for evaluation, thus, length of traing and evaluation datasets was 21 and 7 respectively.  

## Baseline metrics
The initial metrics of the model was:  
```'rouge1': 0.16377685882629334, 'rouge2': 0.046982411354296746, 'rougeL': 0.12755042749961515, 'rougeLsum': 0.12139481949590211```  

## Fine-tune the model
Fine-tuning results are presented below. Due to the low data set size and huge model the overfitting take place. So, the optimal model is near 40-50 steps, the 50th step has been chosen for further testing.
```
Step	Training Loss	Validation Loss	Rouge1	Rouge2	Rougel	Rougelsum	Gen Len
10	2.138800	1.958890	51.038700	19.848700	44.657100	49.208400	512.000000
20	1.774500	1.711860	54.777600	23.708600	49.538500	53.298300	512.000000
30	1.428300	1.553371	55.933100	25.086600	50.577100	54.672000	511.857100
40	1.113900	1.480875	56.834700	25.783800	51.017500	55.496400	512.000000
50	0.820000	1.483367	58.510300	27.532700	52.631100	57.204200	512.000000
60	0.581400	1.566640	59.493300	30.251300	54.205300	58.392500	512.000000
70	0.371100	1.685652	60.163500	30.877200	54.777000	58.914300	512.000000
80	0.251900	1.815350	59.639700	30.000700	53.794700	58.322900	512.000000
90	0.166600	1.914930	59.217000	30.322500	53.536700	57.812100	512.000000
100	0.133500	1.963228	58.802000	29.531800	52.866200	57.425200	512.000000
```

<img width="500" alt="image" src="https://github.com/user-attachments/assets/d6f5e940-f045-4e21-aa99-86c49b5447dd">

## Add google web search
Additionally, web search and including obtained data to the model has been added. Web search extracts only snippets and titles of first 20 records and puts these data in the context of LLM. The result is presented below. Metrics got worse due to additional noise of web search.
```'rouge1': 0.41674781502238, 'rouge2': 0.161282673305801, 'rougeL': 0.28228741048070916, 'rougeLsum': 0.28216720843692855```

# Run demo
### Requirements
* The script has been optimized for AWS EC2 instance g5.xlarge, AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3 (Ubuntu 20.04).
* Open 8001 port in inbound rules in sec. group for the instance
### Run script
* To run script just build docker image by command:
```docker compose up --build -d```  
* Open streamlit application in url: <ec2_url>:8001  

Enjoy!

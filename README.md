# Variational Collective Graph AutoEncoder for Multi-behavior Recommendation (VCGAE)
## Contents
<ul>
  <li style="list-style-type:circle;"> Dataset
  <li style="list-style-type:circle;"> Codes
</ul>

## Environment
The codes of our VCGAE are implemented under the following development environment:
<ul>
  <li style="list-style-type:circle;">python=3.6.9</li>
  <li style="list-style-type:circle;">tensorflow=1.15.3</li>
  <li style="list-style-type:circle;">numpy=1.17.3</li>
  <li style="list-style-type:circle;">scipy=1.3.1</li>
</ul>


## How to Run the Codes
<ul>
  <li style="list-style-type:circle;">JD</li>
</ul>



```python
python   VCGAE.py    --tst_file /tst_buy     --tradeOff=0.05  --tradeOff_cr=0.1 
```

/*
## Partial results
The following table shows the recommendation performance of our VCGAE on JD, Tmall, and UB dataset.

|Algorithms|Dataset|Pre@10|Rec@10|HR@10|NDCG@10|
|:-|:-:|:-:|:-:|:-:|:-:|
|VCGAE|JD|0.0343|0.1444|0.1947|0.1095|
|VCGAE|Tmall|0.0014|0.0065| 0.0131| 0.0069|
|VCGAE|UB|0.0081 | 0.0457 | 0.0670| 0.0487 |
*/


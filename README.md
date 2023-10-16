![image](https://github.com/YMADE8/VCGAE/assets/124166493/f44c4bbd-92ee-4ce0-b5c1-35813f5ee412)![image](https://github.com/YMADE8/VCGAE/assets/124166493/93a821b9-cd74-4147-a921-3e4930050790)# Variational Collective Graph AutoEncoder for Multi-behavior Recommendation (VCGAE)
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
python VCGAE.py --dataset JD --tst_file /tst_buy  --layer_size=[100,100,100]   --lr=0.001  --node_dropout_flag=1  --node_dropout=[0.1]   --mess_dropout=[0.1]   --tradeOff=0.1  --tradeOff_cr=0.1    --Ks=[5,10,15]  
```

<ul>
  <li style="list-style-type:circle;">Tmall</li>
</ul>

```python
python VCGAE.py --dataset=Tmall --n=17202  --m=16177  --tst_file /tst_buy  --layer_size=[100,100,100,100]   --lr=0.001    --node_dropout_flag=1  --node_dropout=[0.1]   --mess_dropout=[0.1]   --tradeOff=0.01  --tradeOff_cr=0.1   --Ks=[5,10,15]

```

<ul>
  <li style="list-style-type:circle;">UB</li>
</ul>

```python
python VCGAE.py --dataset UB  --tst_file /tst_buy  --layer_size=[100,100,100,100]   --lr=0.001    --node_dropout_flag=1  --node_dropout=[0.1]   --mess_dropout=[0.5]   --tradeOff=1  --tradeOff_cr=1  --Ks=[5,10,15] 
```


<!--
## Partial results
The following table shows the recommendation performance of our VCGAE on JD, Tmall, and UB dataset.

|Algorithms|Dataset|Pre@10|Rec@10|HR@10|NDCG@10|
|:-|:-:|:-:|:-:|:-:|:-:|
|VCGAE|JD|0.0343|0.1444|0.1947|0.1095|
|VCGAE|Tmall|0.0014|0.0065| 0.0131| 0.0069|
|VCGAE|UB|0.0081 | 0.0457 | 0.0670| 0.0487 |
-->

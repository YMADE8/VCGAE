● 数据文件的命名方式
1. target: 目标行为，即purchase
2. auxiliary: 所有辅助行为的并集（JD:click+favourite, Tmall: click+favourite, UB: click+ favourite+cart）
3. union: 所有行为的并集，即auxiliary+target，不包含测试集和验证集
4. union2: 两种主要行为的并集，即click+purchase
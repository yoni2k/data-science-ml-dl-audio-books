# Conclusions per execution:
## Conclusions 3 - some starting params - different activation functions
- Train accuracies are around .82%, validate and test around .80% - overfitting
- Training time around 3 seconds for most, 1 model with 7 seconds
- Funcs are different every time. 
    - 4/5 relu first, (other tanh)
    - relu, tanh appears twice in best out of 5
    - All functions appeared at least once 
### Conclusions going forward:
- Need to try with a lot of different params

## Conclusions 4 - smaller learning rate - 0.0001
- Doesn't make much of a difference 
### Conclusions going forward:
- Need to try with a lot of different params

## Conclusions 5 - baseline for relu
- Batch sizes	Hidden Widths	Nums layers	Functions	Learning rates	Improvement deltas	Improvement patience	Improvement restore weights
    [100]	    [100]	        [4]	        ['relu']	[0.0001]	    [0.001]	            [5]                     [True]
- Local run
- Currently concentrating on train loss: ~.81 (.80-.82)
- Time: ~ 5.4 on average 
### Conclusions going forward:
- Use as a baseline
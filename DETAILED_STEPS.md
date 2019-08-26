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
- Batch sizes	Hidden Widths	Nums layers	Functions	        Learning rates	Improvement deltas	Improvement patience	Improvement restore weights
  [450]	        [450]	        [4]         [('relu', 'relu')]	[0.001]	        [0.0001]	        [10]	                [True]
- Local run
- Currently concentrating on train loss: ~.82
- Time: ~ 2.5 on average 
### Conclusions going forward:
- Use as a baseline

## Conclusions 6 - smaller batch size - 200
- Local run
- Currently concentrating on train loss: ~.81.3
- Time: ~ 3.6 on average 
### Conclusions going forward:
- Takes more time, loss is slightly worse, but not conclusive, try higher

## Conclusions 7 - larger batch size - 600
- Local run
- Currently concentrating on train loss: ~.82.8
- Time: ~ 4.7 on average 
### Conclusions going forward:
- Takes more time, but better accuracy, try higher

## Conclusions 8 - larger batch size - 1000
- Local run
- Currently concentrating on train loss: ~.82.7
- Time: ~ 4.5 on average 
### Conclusions going forward:
- Not conclusive if better or worse, try even higher

## Conclusions 9 - larger batch size - 1500
- Local run
- Currently concentrating on train loss: ~.81.5
- Time: ~ 3.8 on average 
### Conclusions going forward:
- Seems going too high with batch size, makes the results worse. Try no batches

## Conclusions 10 - larger batch size - 3580 - no batches
- Local run
- Currently concentrating on train loss: ~.83
- Time: ~ 5 on average 
### Conclusions going forward:
- No batches seems better, stay with it for now, but don't give up batch sizes completely

## Conclusions 11 - width smaller - 300 
- Local run
- Currently concentrating on train loss: ~.83
- Time: ~ 2.5 on average 
### Conclusions going forward:
- Seems that going to smaller width didn't cause worse results, go down more

## Conclusions 12 - width smaller - 200 
- Local run
- Currently concentrating on train loss: ~.827
- Time: ~ 3.2 on average 
### Conclusions going forward:
- Not conclusive, but seems worse both accuracy-wise and time-wise, try even lower

## Conclusions 13 - width smaller - 100 
- Local run
- Currently concentrating on train loss: ~.827
- Time: ~ 2.4 on average 
### Conclusions going forward:
- Not conclusive, but seems there is no difference between 100 and 200, but quicker, go down more

## Conclusions 14 - width smaller - 50 
- Local run
- Currently concentrating on train loss: ~.83
- Time: ~ 4.1 on average 
### Conclusions going forward:
- Not conclusive, but seems there is no difference between 200, 100 and 50, but slower, go down more

## Conclusions 15 - width smaller - 25 
- Local run
- Currently concentrating on train loss: ~.82.1
- Time: ~ 5.2 on average 
### Conclusions going forward:
- Seems 25 width is too small (and slower). Try much higher, above 450 that was tried before

## Conclusions 16 - width larger - 600 
- Local run
- Currently concentrating on train loss: ~.82.8
- Time: ~ 3.5 on average 
### Conclusions going forward:
- Either same or worse than 450. Try even higher

## Conclusions 17 - width larger - 800 
- Local run
- Currently concentrating on train loss: ~.83
- Time: ~ 5.1 on average 
### Conclusions going forward:
- Same as 450, go higher

## Conclusions 18 - width larger - 1200 
- Local run
- Currently concentrating on train loss: ~.83
- Time: ~ 14 on average 
### Conclusions going forward:
- Going even higher than 1000 gives similar results but much slower.  Stay with width 100 - seems fast and good results.  Enough opportunities for connections of 10 inputs

## Conclusions 19 - learning rate 0.0001 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.821
- Time: ~ 12 on average 
### Conclusions going forward:
- Seems worse accuracy, but much slower, try value in between 0.0005

## Conclusions 20 - learning rate 0.0005 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.832
- Time: ~ 4.6 on average 
### Conclusions going forward:
- Seems same or slightly better than 0.001, and slightly faster? Try, 0.0003 and 0.0007

## Conclusions 21 - learning rate 0.0003 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.836
- Time: ~ 6 on average 
### Conclusions going forward:
- Slower but slightly better - try higher rate than 0.001 - 0.0015 

## Conclusions 22 - learning rate 0.0015 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.834
- Time: ~ 2.3 on average 
### Conclusions going forward:
- Much faster, seems not worse, but possibly even better (although probably luck). Try going even higher - 0.003 

## Conclusions 24 - learning rate 0.005 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.825
- Time: ~ 1 on average 
### Conclusions going forward:
- Much faster, slightly worse accuracy-wise, try to go even higher, maybe there is a different minimum 

## Conclusions 25 - learning rate 0.01 instead of default 0.001 
- Local run
- Currently concentrating on train loss: ~.821
- Time: ~ 0.8 on average 
### Conclusions going forward:
- Stay for now with 0.001 that is not as good as 0.0003 but faster, and not much worse

## Conclusions 26 - patience very large and delta very small - continue much more 
- Local run
- Currently concentrating on train loss: ~.827
- Time: ~ 2.7 on average 
### Conclusions going forward:
- Doesn't seem that it's number of epochs that are missing
 

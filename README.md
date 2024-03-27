# <img src=https://raw.githubusercontent.com/aiqm/torchani/master/logo1.png width=180/>  Accurate Neural Network Potential on PyTorch

This version of torchani allows you to scale predictions between a min and max value with by adding a sigmoid to the final output of the combinations of the AEV neural networks.


This version adds a n+1 column to the AEV denoting the charge of the molecule. We are testing if this feature can help the network discern between molecules of different charge without having to train 1 network for Z=0 and another for Z=1.